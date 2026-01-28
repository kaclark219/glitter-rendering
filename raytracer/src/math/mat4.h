#include <cmath>
#include <iostream>

class Mat4 {
    private:
        float data[4][4];

    public:
        // default constructor (identity matrix)
        Mat4() {
            for (int i = 0; i < 4; ++i) {
                for (int j = 0; j < 4; ++j) {
                    data[i][j] = (i == j) ? 1.0f : 0.0f;
                }
            }
        }

        // filled matrix parameter constructor
        Mat4(float d[4][4]) {
            for (int i = 0; i < 4; ++i) {
                for (int j = 0; j < 4; ++j) {
                    data[i][j] = d[i][j];
                }
            }
        }

        // copy constructor
        Mat4(const Mat4 &m) {
            for (int i = 0; i < 4; ++i) {
                for (int j = 0; j < 4; ++j) {
                    data[i][j] = m.data[i][j];
                }
            }
        }

        // getters
        float get(int row, int col) const {
            return data[row][col];
        }

        // setters
        void set(int row, int col, float value) {
            data[row][col] = value;
        }

        // print matrix (for debugging)
        void print() const {
            for (int i = 0; i < 4; ++i) {
                for (int j = 0; j < 4; ++j) {
                    std::cout << data[i][j] << " ";
                }
                std::cout << std::endl;
            }
        }

        // make translation matrix
        static Mat4 translation(float tx, float ty, float tz) {
            Mat4 result;
            result.set(0, 3, tx);
            result.set(1, 3, ty);
            result.set(2, 3, tz);
            return result;
        }

        // make scaling matrix
        static Mat4 scaling(float sx, float sy, float sz) {
            Mat4 result;
            result.set(0, 0, sx);
            result.set(1, 1, sy);
            result.set(2, 2, sz);
            return result;
        }

        // make rotation matrix, specified by axis (x, y, z) and angle in radians
        static Mat4 rotation(float axis_x, float axis_y, float axis_z, float angle) {
            Mat4 result;
            float len = std::sqrt(axis_x * axis_x + axis_y * axis_y + axis_z * axis_z);
            if (len == 0.0f) return result;

            float x = axis_x / len;
            float y = axis_y / len;
            float z = axis_z / len;

            float c = std::cos(angle);
            float s = std::sin(angle);
            float t = 1 - c;

            result.set(0, 0, t * x * x + c);
            result.set(0, 1, t * x * y - s * z);
            result.set(0, 2, t * x * z + s * y);
            result.set(0, 3, 0.0f);
            result.set(1, 0, t * y * x + s * z);
            result.set(1, 1, t * y * y + c);
            result.set(1, 2, t * y * z - s * x);
            result.set(1, 3, 0.0f);
            result.set(2, 0, t * z * x - s * y);
            result.set(2, 1, t * z * y + s * x);
            result.set(2, 2, t * z * z + c);
            result.set(2, 3, 0.0f);
            result.set(3, 0, 0.0f);
            result.set(3, 1, 0.0f);
            result.set(3, 2, 0.0f);
            result.set(3, 3, 1.0f);

            return result;
        }
};