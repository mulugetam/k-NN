#include <immintrin.h>
#include <algorithm>
#include <array>
#include <cstddef>
#include <stdint.h>
#include <cmath>

#include "simd_similarity_function_common.cpp"
#include "faiss_score_to_lucene_transform.cpp"


//
// FP16
//

static inline __m512 cvtbf16_ps(__m256i bf16x16) {
    return _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(bf16x16), 16));
}

template <BulkScoreTransform BulkScoreTransformFunc, ScoreTransform ScoreTransformFunc>
struct AVX512SPRFP16MaxIP final : BaseSimilarityFunction<BulkScoreTransformFunc, ScoreTransformFunc> {
    void calculateSimilarityInBulk(SimdVectorSearchContext* srchContext,
                                   int32_t* internalVectorIds,
                                   float* scores,
                                   const int32_t numVectors) {

        int32_t processedCount = 0;
        const auto* queryPtr = (const float*) srchContext->queryVectorSimdAligned;
        const int32_t dim = srchContext->dimension;

        constexpr int32_t vecBlock      = 8;
        constexpr int32_t elemPerLoad   = 16;
        constexpr int32_t prefetchAhead = 3;

        // Pre-compute the mask-free / masked split point once
        const int32_t fullElems = (dim / elemPerLoad) * elemPerLoad;
        const int32_t tailRem   = dim - fullElems;
        const __mmask16 tailMask = tailRem > 0 ? (__mmask16)((1U << tailRem) - 1) : 0;

        __m512 sum[vecBlock];

        for (; processedCount + vecBlock <= numVectors; processedCount += vecBlock) {
            const uint8_t* vectors[vecBlock];
            srchContext->getVectorPointersInBulk((uint8_t**)vectors, &internalVectorIds[processedCount], vecBlock);

            #pragma unroll
            for (int32_t v = 0; v < vecBlock; ++v) {
                sum[v] = _mm512_setzero_ps();
            }

            // Mask-free hot loop
            for (int32_t i = 0; i < fullElems; i += elemPerLoad) {
                __m512 q0 = _mm512_loadu_ps(queryPtr + i);

                __m512 vRegs[vecBlock];
                #pragma unroll
                for (int32_t v = 0; v < vecBlock; ++v) {
                    vRegs[v] = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*)(vectors[v] + 2 * i)));
                }

                // Prefetch 3 iterations ahead
                const int32_t pfElem = i + prefetchAhead * elemPerLoad;
                if (pfElem < dim) {
                    const int32_t pfByteOffset = pfElem * 2;
                    #pragma unroll
                    for (int32_t v = 0; v < vecBlock; ++v) {
                        __builtin_prefetch(vectors[v] + pfByteOffset, 0, 3);
                    }
                    __builtin_prefetch(queryPtr + pfElem, 0, 3);
                }

                #pragma unroll
                for (int32_t v = 0; v < vecBlock; ++v) {
                    sum[v] = _mm512_fmadd_ps(q0, vRegs[v], sum[v]);
                }
            }

            // Single masked tail (executes 0 or 1 times)
            if (tailRem > 0) {
                __m512 q0 = _mm512_maskz_loadu_ps(tailMask, queryPtr + fullElems);

                __m512 vRegs[vecBlock];
                #pragma unroll
                for (int32_t v = 0; v < vecBlock; ++v) {
                    vRegs[v] = _mm512_cvtph_ps(_mm256_maskz_loadu_epi16(tailMask, vectors[v] + 2 * fullElems));
                }

                #pragma unroll
                for (int32_t v = 0; v < vecBlock; ++v) {
                    sum[v] = _mm512_fmadd_ps(q0, vRegs[v], sum[v]);
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

            for (int32_t i = 0; i < fullElems; i += elemPerLoad) {
                __m512 q = _mm512_loadu_ps(queryPtr + i);
                __m512 v = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*)(vecPtr + 2 * i)));
                sumScalar = _mm512_fmadd_ps(q, v, sumScalar);
            }

            if (tailRem > 0) {
                __m512 q = _mm512_maskz_loadu_ps(tailMask, queryPtr + fullElems);
                __m512 v = _mm512_cvtph_ps(_mm256_maskz_loadu_epi16(tailMask, vecPtr + 2 * fullElems));
                sumScalar = _mm512_fmadd_ps(q, v, sumScalar);
            }

            scores[processedCount] = _mm512_reduce_add_ps(sumScalar);
        }

        BulkScoreTransformFunc(scores, numVectors);
    }
};

template <BulkScoreTransform BulkScoreTransformFunc, ScoreTransform ScoreTransformFunc>
struct AVX512SPRFP16L2 final : BaseSimilarityFunction<BulkScoreTransformFunc, ScoreTransformFunc> {
    void calculateSimilarityInBulk(SimdVectorSearchContext* srchContext,
                                   int32_t* internalVectorIds,
                                   float* scores,
                                   const int32_t numVectors) {

        int32_t processedCount = 0;
        const auto* queryPtr = (const float*) srchContext->queryVectorSimdAligned;
        const int32_t dim = srchContext->dimension;

        constexpr int32_t vecBlock      = 8;
        constexpr int32_t elemPerLoad   = 16;
        constexpr int32_t prefetchAhead = 3;

        const int32_t fullElems = (dim / elemPerLoad) * elemPerLoad;
        const int32_t tailRem   = dim - fullElems;
        const __mmask16 tailMask = tailRem > 0 ? (__mmask16)((1U << tailRem) - 1) : 0;

        __m512 sum[vecBlock];

        for (; processedCount + vecBlock <= numVectors; processedCount += vecBlock) {
            const uint8_t* vectors[vecBlock];
            srchContext->getVectorPointersInBulk((uint8_t**)vectors, &internalVectorIds[processedCount], vecBlock);

            #pragma unroll
            for (int32_t v = 0; v < vecBlock; ++v) {
                sum[v] = _mm512_setzero_ps();
            }

            // Mask-free hot loop
            for (int32_t i = 0; i < fullElems; i += elemPerLoad) {
                __m512 q0 = _mm512_loadu_ps(queryPtr + i);

                __m512 vRegs[vecBlock];
                #pragma unroll
                for (int32_t v = 0; v < vecBlock; ++v) {
                    vRegs[v] = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*)(vectors[v] + 2 * i)));
                }

                // Prefetch 3 iterations ahead
                const int32_t pfElem = i + prefetchAhead * elemPerLoad;
                if (pfElem < dim) {
                    const int32_t pfByteOffset = pfElem * 2;
                    #pragma unroll
                    for (int32_t v = 0; v < vecBlock; ++v) {
                        __builtin_prefetch(vectors[v] + pfByteOffset, 0, 3);
                    }
                    __builtin_prefetch(queryPtr + pfElem, 0, 3);
                }

                #pragma unroll
                for (int32_t v = 0; v < vecBlock; ++v) {
                    __m512 diff = _mm512_sub_ps(q0, vRegs[v]);
                    sum[v] = _mm512_fmadd_ps(diff, diff, sum[v]);
                }
            }

            // Single masked tail
            if (tailRem > 0) {
                __m512 q0 = _mm512_maskz_loadu_ps(tailMask, queryPtr + fullElems);

                __m512 vRegs[vecBlock];
                #pragma unroll
                for (int32_t v = 0; v < vecBlock; ++v) {
                    vRegs[v] = _mm512_cvtph_ps(_mm256_maskz_loadu_epi16(tailMask, vectors[v] + 2 * fullElems));
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

            for (int32_t i = 0; i < fullElems; i += elemPerLoad) {
                __m512 q = _mm512_loadu_ps(queryPtr + i);
                __m512 v = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*)(vecPtr + 2 * i)));
                __m512 diff = _mm512_sub_ps(q, v);
                sumScalar = _mm512_fmadd_ps(diff, diff, sumScalar);
            }

            if (tailRem > 0) {
                __m512 q = _mm512_maskz_loadu_ps(tailMask, queryPtr + fullElems);
                __m512 v = _mm512_cvtph_ps(_mm256_maskz_loadu_epi16(tailMask, vecPtr + 2 * fullElems));
                __m512 diff = _mm512_sub_ps(q, v);
                sumScalar = _mm512_fmadd_ps(diff, diff, sumScalar);
            }

            scores[processedCount] = _mm512_reduce_add_ps(sumScalar);
        }

        BulkScoreTransformFunc(scores, numVectors);
    }
};



//
//  BF16 — New implementation
//
//  All arithmetic is FP32 to avoid accumulation error.
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

        constexpr int32_t vecBlock      = 8;
        constexpr int32_t elemPerLoad   = 16;
        constexpr int32_t prefetchAhead = 3;

        const int32_t fullElems = (dim / elemPerLoad) * elemPerLoad;
        const int32_t tailRem   = dim - fullElems;
        const __mmask16 tailMask = tailRem > 0 ? (__mmask16)((1U << tailRem) - 1) : 0;

        __m512 sum[vecBlock];

        for (; processedCount + vecBlock <= numVectors; processedCount += vecBlock) {
            const uint8_t* vectors[vecBlock];
            srchContext->getVectorPointersInBulk((uint8_t**)vectors, &internalVectorIds[processedCount], vecBlock);

            #pragma unroll
            for (int32_t v = 0; v < vecBlock; ++v) {
                sum[v] = _mm512_setzero_ps();
            }

            // Mask-free hot loop
            for (int32_t i = 0; i < fullElems; i += elemPerLoad) {
                __m512 q0 = _mm512_loadu_ps(queryPtr + i);

                __m512 vRegs[vecBlock];
                #pragma unroll
                for (int32_t v = 0; v < vecBlock; ++v) {
                    vRegs[v] = cvtbf16_ps(_mm256_loadu_si256((const __m256i*)(vectors[v] + 2 * i)));
                }

                const int32_t pfElem = i + prefetchAhead * elemPerLoad;
                if (pfElem < dim) {
                    const int32_t pfByteOffset = pfElem * 2;
                    #pragma unroll
                    for (int32_t v = 0; v < vecBlock; ++v) {
                        __builtin_prefetch(vectors[v] + pfByteOffset, 0, 3);
                    }
                    __builtin_prefetch(queryPtr + pfElem, 0, 3);
                }

                #pragma unroll
                for (int32_t v = 0; v < vecBlock; ++v) {
                    sum[v] = _mm512_fmadd_ps(q0, vRegs[v], sum[v]);
                }
            }

            // Single masked tail
            if (tailRem > 0) {
                __m512 q0 = _mm512_maskz_loadu_ps(tailMask, queryPtr + fullElems);

                __m512 vRegs[vecBlock];
                #pragma unroll
                for (int32_t v = 0; v < vecBlock; ++v) {
                    vRegs[v] = cvtbf16_ps(_mm256_maskz_loadu_epi16(tailMask, vectors[v] + 2 * fullElems));
                }

                #pragma unroll
                for (int32_t v = 0; v < vecBlock; ++v) {
                    sum[v] = _mm512_fmadd_ps(q0, vRegs[v], sum[v]);
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

            for (int32_t i = 0; i < fullElems; i += elemPerLoad) {
                __m512 q = _mm512_loadu_ps(queryPtr + i);
                __m512 v = cvtbf16_ps(_mm256_loadu_si256((const __m256i*)(vecPtr + 2 * i)));
                sumScalar = _mm512_fmadd_ps(q, v, sumScalar);
            }

            if (tailRem > 0) {
                __m512 q = _mm512_maskz_loadu_ps(tailMask, queryPtr + fullElems);
                __m512 v = cvtbf16_ps(_mm256_maskz_loadu_epi16(tailMask, vecPtr + 2 * fullElems));
                sumScalar = _mm512_fmadd_ps(q, v, sumScalar);
            }

            scores[processedCount] = _mm512_reduce_add_ps(sumScalar);
        }

        BulkScoreTransformFunc(scores, numVectors);
    }
};

template <BulkScoreTransform BulkScoreTransformFunc, ScoreTransform ScoreTransformFunc>
struct AVX512BF16L2 final : BaseSimilarityFunction<BulkScoreTransformFunc, ScoreTransformFunc> {
    void calculateSimilarityInBulk(SimdVectorSearchContext* srchContext,
                                   int32_t* internalVectorIds,
                                   float* scores,
                                   const int32_t numVectors) {

        int32_t processedCount = 0;
        const auto* queryPtr = (const float*) srchContext->queryVectorSimdAligned;
        const int32_t dim = srchContext->dimension;

        constexpr int32_t vecBlock      = 8;
        constexpr int32_t elemPerLoad   = 16;
        constexpr int32_t prefetchAhead = 3;

        const int32_t fullElems = (dim / elemPerLoad) * elemPerLoad;
        const int32_t tailRem   = dim - fullElems;
        const __mmask16 tailMask = tailRem > 0 ? (__mmask16)((1U << tailRem) - 1) : 0;

        __m512 sum[vecBlock];

        for (; processedCount + vecBlock <= numVectors; processedCount += vecBlock) {
            const uint8_t* vectors[vecBlock];
            srchContext->getVectorPointersInBulk((uint8_t**)vectors, &internalVectorIds[processedCount], vecBlock);

            #pragma unroll
            for (int32_t v = 0; v < vecBlock; ++v) {
                sum[v] = _mm512_setzero_ps();
            }

            // Mask-free hot loop
            for (int32_t i = 0; i < fullElems; i += elemPerLoad) {
                __m512 q0 = _mm512_loadu_ps(queryPtr + i);

                __m512 vRegs[vecBlock];
                #pragma unroll
                for (int32_t v = 0; v < vecBlock; ++v) {
                    vRegs[v] = cvtbf16_ps(_mm256_loadu_si256((const __m256i*)(vectors[v] + 2 * i)));
                }

                const int32_t pfElem = i + prefetchAhead * elemPerLoad;
                if (pfElem < dim) {
                    const int32_t pfByteOffset = pfElem * 2;
                    #pragma unroll
                    for (int32_t v = 0; v < vecBlock; ++v) {
                        __builtin_prefetch(vectors[v] + pfByteOffset, 0, 3);
                    }
                    __builtin_prefetch(queryPtr + pfElem, 0, 3);
                }

                #pragma unroll
                for (int32_t v = 0; v < vecBlock; ++v) {
                    __m512 diff = _mm512_sub_ps(q0, vRegs[v]);
                    sum[v] = _mm512_fmadd_ps(diff, diff, sum[v]);
                }
            }

            // Single masked tail
            if (tailRem > 0) {
                __m512 q0 = _mm512_maskz_loadu_ps(tailMask, queryPtr + fullElems);

                __m512 vRegs[vecBlock];
                #pragma unroll
                for (int32_t v = 0; v < vecBlock; ++v) {
                    vRegs[v] = cvtbf16_ps(_mm256_maskz_loadu_epi16(tailMask, vectors[v] + 2 * fullElems));
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

            for (int32_t i = 0; i < fullElems; i += elemPerLoad) {
                __m512 q = _mm512_loadu_ps(queryPtr + i);
                __m512 v = cvtbf16_ps(_mm256_loadu_si256((const __m256i*)(vecPtr + 2 * i)));
                __m512 diff = _mm512_sub_ps(q, v);
                sumScalar = _mm512_fmadd_ps(diff, diff, sumScalar);
            }

            if (tailRem > 0) {
                __m512 q = _mm512_maskz_loadu_ps(tailMask, queryPtr + fullElems);
                __m512 v = cvtbf16_ps(_mm256_maskz_loadu_epi16(tailMask, vecPtr + 2 * fullElems));
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
