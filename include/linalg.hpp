#ifndef LINALG_YW_H
#define LINALG_YW_H

namespace
{
}

template <typename T>
T determinant(T *data, int size)
{
    if (size == 1)
    {
        return data[0];
    }
    else if (size == 2)
    {
        /*
            2x2

            | a b |
            | c d |

            ad - bc
        */
        return data[0] * data[3] - data[1] * data[2];
    }
    else
    {
        int subMatSize = size - 1;
        T *subMatData = new T[subMatSize * subMatSize];

        T det = 0;
        T subDet;

        for (int i = 0; i < size; ++i)
        {
            for (int j = 0, c = 0; j < size; ++j)
            {
                if (j == i)
                    continue;

                for (int k = 0; k < subMatSize; ++k)
                {
                    subMatData[k * subMatSize + c] = data[(k + 1) * size + j];
                }

                ++c;
            }

            subDet = data[i] * determinant(subMatData, subMatSize);

            if (i % 2 == 1)
            {
                subDet = -subDet;
            }

            det += subDet;
        }

        delete[] subMatData;
        return det;
    }
}

#endif