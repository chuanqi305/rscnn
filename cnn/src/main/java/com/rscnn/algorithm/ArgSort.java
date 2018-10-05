package com.rscnn.algorithm;

public class ArgSort {
    public static int[] sort(float[] input)
    {
        int h = input.length;
        int[] index = new int[h];

        for (int i = 0; i < h; ++i)
            index[i] = i;

        for (int i = h - 1; i > 0; --i)
        {
            int min = 0;
            for (int j = 1; j <= i; ++j)
                if (input[index[j]] < input[index[min]])
                    min = j;

            int temp = index[i];
            index[i] = index[min];
            index[min] = temp;
        }

        return index;
    }
}
