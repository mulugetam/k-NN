/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.memoryoptsearch;

import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.store.IndexInput;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.memoryoptsearch.faiss.FaissIndexScalarQuantizedFlat;
import org.opensearch.knn.memoryoptsearch.faiss.FaissSection;
import org.opensearch.knn.memoryoptsearch.faiss.reconstruct.FaissBF16Reconstructor;
import org.opensearch.knn.memoryoptsearch.faiss.reconstruct.FaissQuantizerType;

import java.lang.reflect.Field;

import static org.mockito.Mockito.mock;

public class FaissIndexScalarQuantizedFlatBF16Tests extends KNNTestCase {

    public void testGetFloatValues_whenQuantizerTypeIsBF16_thenEnterBF16Branch() throws Exception {
        FaissIndexScalarQuantizedFlat index = new FaissIndexScalarQuantizedFlat();

        // Set quantizerType to QT_BF16
        setField(index, FaissIndexScalarQuantizedFlat.class, "quantizerType", FaissQuantizerType.QT_BF16);

        // Set dimension
        setField(index, "dimension", 4);

        // Set oneVectorByteSize (BF16: 2 bytes per dimension)
        setField(index, FaissIndexScalarQuantizedFlat.class, "oneVectorByteSize", 8L);

        // Set totalNumberOfVectors (inherited from FaissIndex)
        setField(index, "totalNumberOfVectors", 10);

        // Set reconstructor
        FaissBF16Reconstructor reconstructor = new FaissBF16Reconstructor(4, 16);
        setField(index, FaissIndexScalarQuantizedFlat.class, "reconstructor", reconstructor);

        // Set flatVectors with a mock FaissSection
        FaissSection flatVectors = mock(FaissSection.class);
        setField(index, FaissIndexScalarQuantizedFlat.class, "flatVectors", flatVectors);

        // Call getFloatValues with a non-mmap IndexInput (mock)
        IndexInput mockInput = mock(IndexInput.class);
        FloatVectorValues result = index.getFloatValues(mockInput);

        assertNotNull(result);
        assertEquals(4, result.dimension());
        assertEquals(10, result.size());
    }

    private static void setField(Object target, String fieldName, Object value) throws Exception {
        setField(target, target.getClass(), fieldName, value);
    }

    private static void setField(Object target, Class<?> clazz, String fieldName, Object value) throws Exception {
        try {
            Field field = clazz.getDeclaredField(fieldName);
            field.setAccessible(true);
            field.set(target, value);
        } catch (NoSuchFieldException e) {
            // Try superclass
            if (clazz.getSuperclass() != null) {
                setField(target, clazz.getSuperclass(), fieldName, value);
            } else {
                throw e;
            }
        }
    }
}
