/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 *
 * Modifications Copyright OpenSearch Contributors. See
 * GitHub history for details.
 */

#if defined(KNN_HAVE_AVX512_SPR)
    // Sapphire Rapids: currently mirrors the AVX-512 FP32 path for both FP16
    // and BF16. See avx512_spr_simd_similarity_function.cpp for details on
    // how FP16 L2 could be accelerated ~2x with native _mm512_fmadd_ph.
    #include "avx512_spr_simd_similarity_function.cpp"
#elif defined(KNN_HAVE_AVX512)
    #include "avx512_simd_similarity_function.cpp"
#elif defined(KNN_HAVE_ARM_FP16)
    #include "arm_neon_simd_similarity_function.cpp"
#else
    #include "default_simd_similarity_function.cpp"
#endif
