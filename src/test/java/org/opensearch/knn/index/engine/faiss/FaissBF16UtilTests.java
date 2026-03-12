/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.engine.faiss;

import org.opensearch.knn.KNNTestCase;

import static org.opensearch.knn.index.engine.faiss.FaissBF16Util.validateBF16VectorValue;

public class FaissBF16UtilTests extends KNNTestCase {

    public void testValidateBF16VectorValue_withNaN_thenThrowException() {
        expectThrows(IllegalArgumentException.class, () -> validateBF16VectorValue(Float.NaN));
    }

    public void testValidateBF16VectorValue_withPositiveInfinity_thenThrowException() {
        expectThrows(IllegalArgumentException.class, () -> validateBF16VectorValue(Float.POSITIVE_INFINITY));
    }

    public void testValidateBF16VectorValue_withNegativeInfinity_thenThrowException() {
        expectThrows(IllegalArgumentException.class, () -> validateBF16VectorValue(Float.NEGATIVE_INFINITY));
    }

    public void testValidateBF16VectorValue_withFiniteValues_thenSucceed() {
        // Since BF16 has the same exponent range as float32, all finite float values are valid
        validateBF16VectorValue(0.0f);
        validateBF16VectorValue(1.0f);
        validateBF16VectorValue(-1.0f);
        validateBF16VectorValue(65504.0f);
        validateBF16VectorValue(-65504.0f);
        validateBF16VectorValue(Float.MAX_VALUE);
        validateBF16VectorValue(-Float.MAX_VALUE);
        validateBF16VectorValue(Float.MIN_VALUE);
    }

    public void testClipVectorValueToBF16Range_succeed() {
        // Since BF16 has the same range as float32, clipping is a no-op for finite values
        assertEquals(65504.0f, FaissBF16Util.clipVectorValueToBF16Range(65504.0f), 0.0f);
        assertEquals(1000000.89f, FaissBF16Util.clipVectorValueToBF16Range(1000000.89f), 0.0f);
        assertEquals(-65504.0f, FaissBF16Util.clipVectorValueToBF16Range(-65504.0f), 0.0f);
        assertEquals(-1000000.89f, FaissBF16Util.clipVectorValueToBF16Range(-1000000.89f), 0.0f);
    }
}
