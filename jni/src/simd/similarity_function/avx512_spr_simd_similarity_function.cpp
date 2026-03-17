#include <immintrin.h>
#include <algorithm>
#include <array>
#include <cstddef>
#include <stdint.h>
#include <cmath>

#include "simd_similarity_function_common.cpp"
#include "faiss_score_to_lucene_transform.cpp"

// BF16 -> FP32: zero-extend 16 -> 32 bits then shift left by 16
static inline __m512 cvtbf16_ps(__m256i bf16x16) {
    return _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(bf16x16), 16));
}

// FP32 -> BF16 SIMD (round-to-nearest-even)
static inline __m256i cvtps_bf16(__m512 f) {
    __m512i fi = _mm512_castps_si512(f);
    __m512i lsb = _mm512_and_si512(_mm512_srli_epi32(fi, 16), _mm512_set1_epi32(1));
    __m512i rounded = _mm512_add_epi32(_mm512_add_epi32(fi, _mm512_set1_epi32(0x7FFF)), lsb);
    return _mm512_cvtepi32_epi16(_mm512_srli_epi32(rounded, 16));
}

// Number of FP16 accumulations before draining to FP32 to limit precision loss.
// Smaller -> more accurate, larger -> faster.
static constexpr int32_t FP16_DRAIN_INTERVAL = 4;

// Drain FP16 accumulator into FP32 accumulator, then zero the FP16 accum.
static inline void drain_ph_to_ps(__m512& acc_ps, __m512h& acc_ph) {
    __m512i raw = _mm512_castph_si512(acc_ph);
    acc_ps = _mm512_add_ps(acc_ps,
                _mm512_cvtph_ps(_mm512_castsi512_si256(raw)));
    acc_ps = _mm512_add_ps(acc_ps,
                _mm512_cvtph_ps(_mm512_extracti64x4_epi64(raw, 1)));
    acc_ph = _mm512_castsi512_ph(_mm512_setzero_si512());
}

// Convert 32 consecutive FP32 values -> 32 FP16 values
static inline __m512h cvt2x16_fp32_to_ph(const float* p) {
    __m256i lo = _mm512_cvtps_ph(_mm512_loadu_ps(p),
                    _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    __m256i hi = _mm512_cvtps_ph(_mm512_loadu_ps(p + 16),
                    _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    return _mm512_castsi512_ph(
        _mm512_inserti64x4(_mm512_castsi256_si512(lo), hi, 1));
}

// Convert 32 consecutive FP32 values -> 32 BF16 values
static inline __m512bh cvt2x16_fp32_to_bf16(const float* p) {
    __m256i lo = cvtps_bf16(_mm512_loadu_ps(p));
    __m256i hi = cvtps_bf16(_mm512_loadu_ps(p + 16));
    return (__m512bh)_mm512_inserti64x4(_mm512_castsi256_si512(lo), hi, 1);
}

// 
// FP16 IP — Native AVX512-FP16. FP32 query -> FP16
//
//
template <BulkScoreTransform BulkScoreTransformFunc, ScoreTransform ScoreTransformFunc>
struct AVX512SPRFP16MaxIP final : BaseSimilarityFunction<BulkScoreTransformFunc, ScoreTransformFunc> {
    void calculateSimilarityInBulk(SimdVectorSearchContext* srchContext,
                                   int32_t* internalVectorIds,
                                   float* scores,
                                   const int32_t numVectors) {

        int32_t processedCount = 0;
        const auto* queryPtr = (const float*) srchContext->queryVectorSimdAligned;
        const int32_t dim = srchContext->dimension;

        constexpr int32_t vecBlock = 8;
        constexpr int32_t elemPerLoad = 32;

        // SIMD-aligned dim and tail dim
        const int32_t simdDim = (dim / elemPerLoad) * elemPerLoad;
        const int32_t tailDim  = dim - simdDim;

        // Precompute tail mask
        const __mmask32 tailMask = tailDim > 0 ? (__mmask32)((1ULL << tailDim) - 1) : 0;

        // The tail part
        const int32_t tailMid16 = (tailDim >= 16) ? 16 : 0;
        const int32_t tailFinal = tailDim - tailMid16;
        const __mmask16 tailFinalMask = tailFinal > 0 ? (__mmask16)((1U << tailFinal) - 1) : 0;

        __m512h sumFp16[vecBlock]; // FP16 accumulators (32 lanes each)
        __m512  sumFp32[vecBlock]; // FP32 accumulators (16 lanes each)

        for (; processedCount <= numVectors - vecBlock; processedCount += vecBlock) {
            const uint8_t* vectors[vecBlock];
            srchContext->getVectorPointersInBulk((uint8_t**)vectors, &internalVectorIds[processedCount], vecBlock);

            #pragma unroll
            for (int32_t v = 0; v < vecBlock; ++v) {
                sumFp16[v] = _mm512_castsi512_ph(_mm512_setzero_si512());
                sumFp32[v] = _mm512_setzero_ps();
            }

            int32_t drainCount = 0;

            // Mask-free hot loop
            for (int32_t i = 0; i < simdDim; i += elemPerLoad) {
                __m512h q = cvt2x16_fp32_to_ph(queryPtr + i);

                if ((i + elemPerLoad) < dim) {
                    const int32_t nextByteOffset = (i + elemPerLoad) * 2;
                    #pragma unroll
                    for (int32_t v = 0; v < vecBlock; ++v) {
                        __builtin_prefetch(vectors[v] + nextByteOffset, 0, 3);
                    }
                    __builtin_prefetch(queryPtr + i + elemPerLoad, 0, 3);
                }

                #pragma unroll
                for (int32_t v = 0; v < vecBlock; ++v) {
                    __m512h vec = _mm512_loadu_ph(vectors[v] + 2 * i);
                    sumFp16[v] = _mm512_fmadd_ph(q, vec, sumFp16[v]);
                }

                if (++drainCount >= FP16_DRAIN_INTERVAL) {
                    #pragma unroll
                    for (int32_t v = 0; v < vecBlock; ++v)
                        drain_ph_to_ps(sumFp32[v], sumFp16[v]);
                    drainCount = 0;
                }
            }

            // Single masked tail
            if (tailDim > 0) {
                __m256i qLoH, qHiH;
                if (tailMid16 > 0) {
                    qLoH = _mm512_cvtps_ph(_mm512_loadu_ps(queryPtr + simdDim),
                                _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
                    qHiH = tailFinal > 0
                        ? _mm512_cvtps_ph(_mm512_maskz_loadu_ps(tailFinalMask, queryPtr + simdDim + 16),
                                _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC)
                        : _mm256_setzero_si256();
                } else {
                    qLoH = _mm512_cvtps_ph(_mm512_maskz_loadu_ps(tailFinalMask, queryPtr + simdDim),
                                _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
                    qHiH = _mm256_setzero_si256();
                }
                __m512h q = _mm512_castsi512_ph(
                    _mm512_inserti64x4(_mm512_castsi256_si512(qLoH), qHiH, 1));

                #pragma unroll
                for (int32_t v = 0; v < vecBlock; ++v) {
                    __m512h vec = _mm512_castsi512_ph(
                        _mm512_maskz_loadu_epi16(tailMask, vectors[v] + 2 * simdDim));
                    sumFp16[v] = _mm512_fmadd_ph(q, vec, sumFp16[v]);
                }
            }

            #pragma unroll
            for (int32_t v = 0; v < vecBlock; ++v) {
                drain_ph_to_ps(sumFp32[v], sumFp16[v]);
                scores[processedCount + v] = _mm512_reduce_add_ps(sumFp32[v]);
            }
        }

        // Tail loop for remaining vectors
        for (; processedCount < numVectors; ++processedCount) {
            const auto* vecPtr = (const uint8_t*) srchContext->getVectorPointer(internalVectorIds[processedCount]);
            __m512h sumFp16 = _mm512_castsi512_ph(_mm512_setzero_si512());
            __m512  sumFp32 = _mm512_setzero_ps();
            int32_t drainCount = 0;

            for (int32_t i = 0; i < simdDim; i += elemPerLoad) {
                sumFp16 = _mm512_fmadd_ph(cvt2x16_fp32_to_ph(queryPtr + i),
                                           _mm512_loadu_ph(vecPtr + 2 * i), sumFp16);
                if (++drainCount >= FP16_DRAIN_INTERVAL) {
                    drain_ph_to_ps(sumFp32, sumFp16);
                    drainCount = 0;
                }
            }

            if (tailDim > 0) {
                __m256i qLoH, qHiH;
                if (tailMid16 > 0) {
                    qLoH = _mm512_cvtps_ph(_mm512_loadu_ps(queryPtr + simdDim),
                                _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
                    qHiH = tailFinal > 0
                        ? _mm512_cvtps_ph(_mm512_maskz_loadu_ps(tailFinalMask, queryPtr + simdDim + 16),
                                _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC)
                        : _mm256_setzero_si256();
                } else {
                    qLoH = _mm512_cvtps_ph(_mm512_maskz_loadu_ps(tailFinalMask, queryPtr + simdDim),
                                _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
                    qHiH = _mm256_setzero_si256();
                }
                __m512h q = _mm512_castsi512_ph(
                    _mm512_inserti64x4(_mm512_castsi256_si512(qLoH), qHiH, 1));
                sumFp16 = _mm512_fmadd_ph(q,
                    _mm512_castsi512_ph(_mm512_maskz_loadu_epi16(tailMask, vecPtr + 2 * simdDim)),
                    sumFp16);
            }

            drain_ph_to_ps(sumFp32, sumFp16);
            scores[processedCount] = _mm512_reduce_add_ps(sumFp32);
        }

        BulkScoreTransformFunc(scores, numVectors);
    }
};

//
// FP16 L2 — Native AVX512-FP16. FP32 query -> FP16.
//
//
template <BulkScoreTransform BulkScoreTransformFunc, ScoreTransform ScoreTransformFunc>
struct AVX512SPRFP16L2 final : BaseSimilarityFunction<BulkScoreTransformFunc, ScoreTransformFunc> {
    void calculateSimilarityInBulk(SimdVectorSearchContext* srchContext,
                                   int32_t* internalVectorIds,
                                   float* scores,
                                   const int32_t numVectors) {

        int32_t processedCount = 0;
        const auto* queryPtr = (const float*) srchContext->queryVectorSimdAligned;
        const int32_t dim = srchContext->dimension;

        constexpr int32_t vecBlock = 8;
        constexpr int32_t elemPerLoad = 32;

        // SIMD-aligned dim and tail dim
        const int32_t simdDim = (dim / elemPerLoad) * elemPerLoad;
        const int32_t tailDim = dim - simdDim;

        // Precompute tail mask
        const __mmask32 tailMask = tailDim > 0 ? (__mmask32)((1ULL << tailDim) - 1) : 0;

        // The tail part
        const int32_t tailMid16 = (tailDim >= 16) ? 16 : 0;
        const int32_t tailFinal = tailDim - tailMid16;
        const __mmask16 tailFinalMask = tailFinal > 0 ? (__mmask16)((1U << tailFinal) - 1) : 0;

        __m512h sumFp16[vecBlock]; // FP16 accumulators (32 lanes each)
        __m512  sumFp32[vecBlock]; // FP32 accumulators (16 lanes each)

        for (; processedCount <= numVectors - vecBlock; processedCount += vecBlock) {
            const uint8_t* vectors[vecBlock];
            srchContext->getVectorPointersInBulk((uint8_t**)vectors, &internalVectorIds[processedCount], vecBlock);

            #pragma unroll
            for (int32_t v = 0; v < vecBlock; ++v) {
                sumFp16[v] = _mm512_castsi512_ph(_mm512_setzero_si512());
                sumFp32[v] = _mm512_setzero_ps();
            }

            int32_t drainCount = 0;

            // Mask-free hot loop
            for (int32_t i = 0; i < simdDim; i += elemPerLoad) {
                __m512h q = cvt2x16_fp32_to_ph(queryPtr + i);

                if ((i + elemPerLoad) < dim) {
                    const int32_t nextByteOffset = (i + elemPerLoad) * 2;
                    #pragma unroll
                    for (int32_t v = 0; v < vecBlock; ++v) {
                        __builtin_prefetch(vectors[v] + nextByteOffset, 0, 3);
                    }
                    __builtin_prefetch(queryPtr + i + elemPerLoad, 0, 3);
                }

                #pragma unroll
                for (int32_t v = 0; v < vecBlock; ++v) {
                    __m512h vec = _mm512_loadu_ph(vectors[v] + 2 * i);
                    __m512h diff = _mm512_sub_ph(q, vec);
                    sumFp16[v] = _mm512_fmadd_ph(diff, diff, sumFp16[v]);
                }

                if (++drainCount >= FP16_DRAIN_INTERVAL) {
                    #pragma unroll
                    for (int32_t v = 0; v < vecBlock; ++v)
                        drain_ph_to_ps(sumFp32[v], sumFp16[v]);
                    drainCount = 0;
                }
            }

            // Single masked tail
            if (tailDim > 0) {
                __m256i qLoH, qHiH;
                if (tailMid16 > 0) {
                    qLoH = _mm512_cvtps_ph(_mm512_loadu_ps(queryPtr + simdDim),
                                _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
                    qHiH = tailFinal > 0
                        ? _mm512_cvtps_ph(_mm512_maskz_loadu_ps(tailFinalMask, queryPtr + simdDim + 16),
                                _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC)
                        : _mm256_setzero_si256();
                } else {
                    qLoH = _mm512_cvtps_ph(_mm512_maskz_loadu_ps(tailFinalMask, queryPtr + simdDim),
                                _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
                    qHiH = _mm256_setzero_si256();
                }
                __m512h q = _mm512_castsi512_ph(
                    _mm512_inserti64x4(_mm512_castsi256_si512(qLoH), qHiH, 1));

                #pragma unroll
                for (int32_t v = 0; v < vecBlock; ++v) {
                    __m512h vec = _mm512_castsi512_ph(
                        _mm512_maskz_loadu_epi16(tailMask, vectors[v] + 2 * simdDim));
                    __m512h diff = _mm512_sub_ph(q, vec);
                    sumFp16[v] = _mm512_fmadd_ph(diff, diff, sumFp16[v]);
                }
            }

            #pragma unroll
            for (int32_t v = 0; v < vecBlock; ++v) {
                drain_ph_to_ps(sumFp32[v], sumFp16[v]);
                scores[processedCount + v] = _mm512_reduce_add_ps(sumFp32[v]);
            }
        }

        // Tail loop for remaining vectors
        for (; processedCount < numVectors; ++processedCount) {
            const auto* vecPtr = (const uint8_t*) srchContext->getVectorPointer(internalVectorIds[processedCount]);
            __m512h sumFp16 = _mm512_castsi512_ph(_mm512_setzero_si512());
            __m512  sumFp32 = _mm512_setzero_ps();
            int32_t drainCount = 0;

            for (int32_t i = 0; i < simdDim; i += elemPerLoad) {
                __m512h diff = _mm512_sub_ph(cvt2x16_fp32_to_ph(queryPtr + i),
                                             _mm512_loadu_ph(vecPtr + 2 * i));
                sumFp16 = _mm512_fmadd_ph(diff, diff, sumFp16);
                if (++drainCount >= FP16_DRAIN_INTERVAL) {
                    drain_ph_to_ps(sumFp32, sumFp16);
                    drainCount = 0;
                }
            }

            if (tailDim > 0) {
                __m256i qLoH, qHiH;
                if (tailMid16 > 0) {
                    qLoH = _mm512_cvtps_ph(_mm512_loadu_ps(queryPtr + simdDim),
                                _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
                    qHiH = tailFinal > 0
                        ? _mm512_cvtps_ph(_mm512_maskz_loadu_ps(tailFinalMask, queryPtr + simdDim + 16),
                                _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC)
                        : _mm256_setzero_si256();
                } else {
                    qLoH = _mm512_cvtps_ph(_mm512_maskz_loadu_ps(tailFinalMask, queryPtr + simdDim),
                                _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
                    qHiH = _mm256_setzero_si256();
                }
                __m512h q = _mm512_castsi512_ph(
                    _mm512_inserti64x4(_mm512_castsi256_si512(qLoH), qHiH, 1));
                __m512h diff = _mm512_sub_ph(q,
                    _mm512_castsi512_ph(_mm512_maskz_loadu_epi16(tailMask, vecPtr + 2 * simdDim)));
                sumFp16 = _mm512_fmadd_ph(diff, diff, sumFp16);
            }

            drain_ph_to_ps(sumFp32, sumFp16);
            scores[processedCount] = _mm512_reduce_add_ps(sumFp32);
        }

        BulkScoreTransformFunc(scores, numVectors);
    }
};

//
// BF16 IP — Native AVX512-BF16. FP32 query -> BF16
//
//
template <BulkScoreTransform BulkScoreTransformFunc, ScoreTransform ScoreTransformFunc>
struct AVX512BF16MaxIP final : BaseSimilarityFunction<BulkScoreTransformFunc, ScoreTransformFunc> {
    void calculateSimilarityInBulk(SimdVectorSearchContext* srchContext,
                                   int32_t* internalVectorIds,
                                   float* scores,
                                   const int32_t numVectors) {

        int32_t processedCount = 0;
        const auto* queryPtr = (const float*) srchContext->queryVectorSimdAligned;
        const int32_t dim = srchContext->dimension;

        constexpr int32_t vecBlock = 8;
        constexpr int32_t elemPerLoad = 32;

        // SIMD-aligned dim and tail dim
        const int32_t simdDim = (dim / elemPerLoad) * elemPerLoad;
        const int32_t tailDim = dim - simdDim;

        // Precompute tail mask
        const __mmask32 tailMask = tailDim > 0 ? (__mmask32)((1ULL << tailDim) - 1) : 0;

        // Do same for the tail part
        const int32_t tailMid16 = (tailDim >= 16) ? 16 : 0;
        const int32_t tailFinal = tailDim - tailMid16;
        const __mmask16 tailFinalMask = tailFinal > 0 ? (__mmask16)((1U << tailFinal) - 1) : 0;

        __m512 sum[vecBlock];

        for (; processedCount <= numVectors - vecBlock; processedCount += vecBlock) {
            const uint8_t* vectors[vecBlock];
            srchContext->getVectorPointersInBulk((uint8_t**)vectors, &internalVectorIds[processedCount], vecBlock);

            #pragma unroll
            for (int32_t v = 0; v < vecBlock; ++v) {
                sum[v] = _mm512_setzero_ps();
            }

            // Mask-free hot loop
            for (int32_t i = 0; i < simdDim; i += elemPerLoad) {
                __m512bh q = cvt2x16_fp32_to_bf16(queryPtr + i);

                if ((i + elemPerLoad) < dim) {
                    const int32_t nextByteOffset = (i + elemPerLoad) * 2;
                    #pragma unroll
                    for (int32_t v = 0; v < vecBlock; ++v) {
                        __builtin_prefetch(vectors[v] + nextByteOffset, 0, 3);
                    }
                    __builtin_prefetch(queryPtr + i + elemPerLoad, 0, 3);
                }

                #pragma unroll
                for (int32_t v = 0; v < vecBlock; ++v) {
                    __m512bh vec = (__m512bh)_mm512_loadu_si512(vectors[v] + 2 * i);
                    sum[v] = _mm512_dpbf16_ps(sum[v], q, vec);
                }
            }

            // Single masked tail
            if (tailDim > 0) {
                __m256i qLoB, qHiB;
                if (tailMid16 > 0) {
                    qLoB = cvtps_bf16(_mm512_loadu_ps(queryPtr + simdDim));
                    qHiB = tailFinal > 0
                        ? cvtps_bf16(_mm512_maskz_loadu_ps(tailFinalMask, queryPtr + simdDim + 16))
                        : _mm256_setzero_si256();
                } else {
                    qLoB = cvtps_bf16(_mm512_maskz_loadu_ps(tailFinalMask, queryPtr + simdDim));
                    qHiB = _mm256_setzero_si256();
                }
                __m512bh q = (__m512bh)_mm512_inserti64x4(
                    _mm512_castsi256_si512(qLoB), qHiB, 1);

                #pragma unroll
                for (int32_t v = 0; v < vecBlock; ++v) {
                    __m512bh vec = (__m512bh)_mm512_maskz_loadu_epi16(
                        tailMask, vectors[v] + 2 * simdDim);
                    sum[v] = _mm512_dpbf16_ps(sum[v], q, vec);
                }
            }

            #pragma unroll
            for (int32_t v = 0; v < vecBlock; ++v) {
                scores[processedCount + v] = _mm512_reduce_add_ps(sum[v]);
            }
        }

        // Tail loop for remaining vectors
        for (; processedCount < numVectors; ++processedCount) {
            const auto* vecPtr = (const uint8_t*) srchContext->getVectorPointer(internalVectorIds[processedCount]);
            __m512 s = _mm512_setzero_ps();

            for (int32_t i = 0; i < simdDim; i += elemPerLoad) {
                s = _mm512_dpbf16_ps(s, cvt2x16_fp32_to_bf16(queryPtr + i),
                                        (__m512bh)_mm512_loadu_si512(vecPtr + 2 * i));
            }

            if (tailDim > 0) {
                __m256i qLoB, qHiB;
                if (tailMid16 > 0) {
                    qLoB = cvtps_bf16(_mm512_loadu_ps(queryPtr + simdDim));
                    qHiB = tailFinal > 0
                        ? cvtps_bf16(_mm512_maskz_loadu_ps(tailFinalMask, queryPtr + simdDim + 16))
                        : _mm256_setzero_si256();
                } else {
                    qLoB = cvtps_bf16(_mm512_maskz_loadu_ps(tailFinalMask, queryPtr + simdDim));
                    qHiB = _mm256_setzero_si256();
                }
                __m512bh q = (__m512bh)_mm512_inserti64x4(
                    _mm512_castsi256_si512(qLoB), qHiB, 1);
                s = _mm512_dpbf16_ps(s, q,
                    (__m512bh)_mm512_maskz_loadu_epi16(tailMask, vecPtr + 2 * simdDim));
            }

            scores[processedCount] = _mm512_reduce_add_ps(s);
        }

        BulkScoreTransformFunc(scores, numVectors);
    }
};

//
// BF16 L2
//
// BF16 has no native subtract for SPR (L2 using dot products didn't give better performance).
//
template <BulkScoreTransform BulkScoreTransformFunc, ScoreTransform ScoreTransformFunc>
struct AVX512BF16L2 final : BaseSimilarityFunction<BulkScoreTransformFunc, ScoreTransformFunc> {
    void calculateSimilarityInBulk(SimdVectorSearchContext* srchContext,
                                   int32_t* internalVectorIds,
                                   float* scores,
                                   const int32_t numVectors) {

        int32_t processedCount = 0;
        const auto* queryPtr = (const float*) srchContext->queryVectorSimdAligned;
        const int32_t dim = srchContext->dimension;

        constexpr int32_t vecBlock = 8;
        constexpr int32_t elemPerLoad = 16;

        // SIMD-aligned dim and tail dim
        const int32_t simdDim = (dim / elemPerLoad) * elemPerLoad;
        const int32_t tailDim   = dim - simdDim;

        // Precompute tail mask
        const __mmask16 tailMask = tailDim > 0 ? (__mmask16)((1U << tailDim) - 1) : 0;

        __m512 sum[vecBlock];

        for (; processedCount <= numVectors - vecBlock; processedCount += vecBlock) {
            const uint8_t* vectors[vecBlock];
            srchContext->getVectorPointersInBulk((uint8_t**)vectors, &internalVectorIds[processedCount], vecBlock);

            #pragma unroll
            for (int32_t v = 0; v < vecBlock; ++v) {
                sum[v] = _mm512_setzero_ps();
            }

            // Mask-free hot loop
            for (int32_t i = 0; i < simdDim; i += elemPerLoad) {
                __m512 q0 = _mm512_loadu_ps(queryPtr + i);

                __m512 vRegs[vecBlock];
                #pragma unroll
                for (int32_t v = 0; v < vecBlock; ++v) {
                    vRegs[v] = cvtbf16_ps(_mm256_loadu_si256((const __m256i*)(vectors[v] + 2 * i)));
                }

                if ((i + elemPerLoad) < dim) {
                    const int32_t nextByteOffset = (i + elemPerLoad) * 2;
                    #pragma unroll
                    for (int32_t v = 0; v < vecBlock; ++v) {
                        __builtin_prefetch(vectors[v] + nextByteOffset, 0, 3);
                    }
                    __builtin_prefetch(queryPtr + i + elemPerLoad, 0, 3);
                }

                #pragma unroll
                for (int32_t v = 0; v < vecBlock; ++v) {
                    __m512 diff = _mm512_sub_ps(q0, vRegs[v]);
                    sum[v] = _mm512_fmadd_ps(diff, diff, sum[v]);
                }
            }

            // Single masked tail
            if (tailDim > 0) {
                __m512 q0 = _mm512_maskz_loadu_ps(tailMask, queryPtr + simdDim);

                __m512 vRegs[vecBlock];
                #pragma unroll
                for (int32_t v = 0; v < vecBlock; ++v) {
                    vRegs[v] = cvtbf16_ps(_mm256_maskz_loadu_epi16(tailMask, vectors[v] + 2 * simdDim));
                }

                #pragma unroll
                for (int32_t v = 0; v < vecBlock; ++v) {
                    __m512 diff = _mm512_sub_ps(q0, vRegs[v]);
                    sum[v] = _mm512_fmadd_ps(diff, diff, sum[v]);
                }
            }

            #pragma unroll
            for (int32_t v = 0; v < vecBlock; ++v) {
                scores[processedCount + v] = _mm512_reduce_add_ps(sum[v]);
            }
        }

        // Tail loop for remaining vectors
        for (; processedCount < numVectors; ++processedCount) {
            const auto* vecPtr = (const uint8_t*) srchContext->getVectorPointer(internalVectorIds[processedCount]);
            __m512 sumScalar = _mm512_setzero_ps();

            for (int32_t i = 0; i < simdDim; i += elemPerLoad) {
                __m512 q = _mm512_loadu_ps(queryPtr + i);
                __m512 v = cvtbf16_ps(_mm256_loadu_si256((const __m256i*)(vecPtr + 2 * i)));
                __m512 diff = _mm512_sub_ps(q, v);
                sumScalar = _mm512_fmadd_ps(diff, diff, sumScalar);
            }

            if (tailDim > 0) {
                __m512 q = _mm512_maskz_loadu_ps(tailMask, queryPtr + simdDim);
                __m512 v = cvtbf16_ps(_mm256_maskz_loadu_epi16(tailMask, vecPtr + 2 * simdDim));
                __m512 diff = _mm512_sub_ps(q, v);
                sumScalar = _mm512_fmadd_ps(diff, diff, sumScalar);
            }

            scores[processedCount] = _mm512_reduce_add_ps(sumScalar);
        }

        BulkScoreTransformFunc(scores, numVectors);
    }
};

//
// FP16
//
// 1. Max IP
AVX512SPRFP16MaxIP<FaissScoreToLuceneScoreTransform::ipToMaxIpTransformBulk, FaissScoreToLuceneScoreTransform::ipToMaxIpTransform> FP16_MAX_INNER_PRODUCT_SIMIL_FUNC;
// 2. L2
AVX512SPRFP16L2<FaissScoreToLuceneScoreTransform::l2TransformBulk, FaissScoreToLuceneScoreTransform::l2Transform> FP16_L2_SIMIL_FUNC;

//
// BF16
//
// 1. Max IP
AVX512BF16MaxIP<FaissScoreToLuceneScoreTransform::ipToMaxIpTransformBulk, FaissScoreToLuceneScoreTransform::ipToMaxIpTransform> BF16_MAX_INNER_PRODUCT_SIMIL_FUNC;
// 2. L2
AVX512BF16L2<FaissScoreToLuceneScoreTransform::l2TransformBulk, FaissScoreToLuceneScoreTransform::l2Transform> BF16_L2_SIMIL_FUNC;

#ifndef __NO_SELECT_FUNCTION
SimilarityFunction* SimilarityFunction::selectSimilarityFunction(const NativeSimilarityFunctionType nativeFunctionType) {
    if (nativeFunctionType == NativeSimilarityFunctionType::FP16_MAXIMUM_INNER_PRODUCT) {
        return &FP16_MAX_INNER_PRODUCT_SIMIL_FUNC;
    } else if (nativeFunctionType == NativeSimilarityFunctionType::FP16_L2) {
        return &FP16_L2_SIMIL_FUNC;
    } else if (nativeFunctionType == NativeSimilarityFunctionType::BF16_MAXIMUM_INNER_PRODUCT) {
        return &BF16_MAX_INNER_PRODUCT_SIMIL_FUNC;
    } else if (nativeFunctionType == NativeSimilarityFunctionType::BF16_L2) {
        return &BF16_L2_SIMIL_FUNC;
    }

    throw std::runtime_error("Invalid native similarity function type was given, nativeFunctionType="
                             + std::to_string(static_cast<int32_t>(nativeFunctionType)));
}
#endif
