/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.memoryoptsearch;

import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.memoryoptsearch.faiss.reconstruct.FaissBF16Reconstructor;

import java.lang.reflect.Field;

public class FaissBF16ReconstructorTests extends KNNTestCase {

    public void testReconstruct_basicValues() {
        int dimension = 4;
        FaissBF16Reconstructor reconstructor = new FaissBF16Reconstructor(dimension, 16);

        // BF16 stores the upper 16 bits of a float32.
        float[] testValues = { 1.0f, -1.0f, 0.0f, 2.0f };
        byte[] quantizedBytes = new byte[dimension * 2];

        // Encode float values as BF16 (little-endian)
        for (int i = 0; i < dimension; i++) {
            int floatBits = Float.floatToIntBits(testValues[i]);
            int bf16Bits = (floatBits >> 16) & 0xFFFF;
            // Little-endian encoding
            quantizedBytes[i * 2] = (byte) (bf16Bits & 0xFF);
            quantizedBytes[i * 2 + 1] = (byte) ((bf16Bits >> 8) & 0xFF);
        }

        float[] result = new float[dimension];
        reconstructor.reconstruct(quantizedBytes, result);

        // BF16 reconstruction should produce exact values for these simple floats
        // (since they don't lose precision in the truncation)
        for (int i = 0; i < dimension; i++) {
            assertEquals("Mismatch at index " + i, testValues[i], result[i], 0.0f);
        }
    }

    public void testReconstruct_precisionLoss() {
        int dimension = 1;
        FaissBF16Reconstructor reconstructor = new FaissBF16Reconstructor(dimension, 16);

        float original = 1.1f;
        int floatBits = Float.floatToIntBits(original);
        int bf16Bits = (floatBits >> 16) & 0xFFFF;

        byte[] quantizedBytes = new byte[2];
        quantizedBytes[0] = (byte) (bf16Bits & 0xFF);
        quantizedBytes[1] = (byte) ((bf16Bits >> 8) & 0xFF);

        float[] result = new float[dimension];
        reconstructor.reconstruct(quantizedBytes, result);

        // The result should be close but not exact due to BF16 precision loss
        float expected = Float.intBitsToFloat(bf16Bits << 16);
        assertEquals(expected, result[0], 0.0f);

        // Should be within ~1% of original
        assertEquals(original, result[0], 0.02f);
    }

    public void testReconstruct_bigEndian() throws Exception {
        int dimension = 4;
        FaissBF16Reconstructor reconstructor = new FaissBF16Reconstructor(dimension, 16);

        Field isLittleEndianField = FaissBF16Reconstructor.class.getDeclaredField("isLittleEndian");
        isLittleEndianField.setAccessible(true);
        isLittleEndianField.setBoolean(reconstructor, false);

        float[] testValues = { 1.0f, -1.0f, 0.0f, 3.5f };
        byte[] quantizedBytes = new byte[dimension * 2];

        // Encode float values as BF16 in big-endian byte order
        for (int i = 0; i < dimension; i++) {
            int floatBits = Float.floatToIntBits(testValues[i]);
            int bf16Bits = (floatBits >> 16) & 0xFFFF;
            // Big-endian encoding: high byte first
            quantizedBytes[i * 2] = (byte) ((bf16Bits >> 8) & 0xFF);
            quantizedBytes[i * 2 + 1] = (byte) (bf16Bits & 0xFF);
        }

        float[] result = new float[dimension];
        reconstructor.reconstruct(quantizedBytes, result);

        for (int i = 0; i < dimension; i++) {
            assertEquals("Mismatch at index " + i, testValues[i], result[i], 0.0f);
        }
    }
}
