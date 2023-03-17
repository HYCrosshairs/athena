#include <stdio.h>
#include <math.h>
#include <GL/glut.h>

#include <iostream>

#define WIDTH 800
#define HEIGHT 600

__global__ void computeCosine(float *d_output, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
    {
        float x = 2.0f * M_PI * i / (float)N;
        d_output[i] = cosf(x);
    }
}

int main(int argc, char **argv)
{
    const int N = 1000;
    float *h_output = new float[N];
    float *d_output;

    cudaMalloc((void **)&d_output, N * sizeof(float));

    int threadsPerBlock = 25;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    computeCosine<<<blocksPerGrid, threadsPerBlock>>>(d_output, N);

    cudaMemcpy(h_output, d_output, N * sizeof(float), cudaMemcpyDeviceToHost);

    for(int i = 0; i < 1000; i++)
    {
        std::cout << d_output[i] << std::endl;
    }

    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_SINGLE | GLUT_RGB);
    glutInitWindowSize(WIDTH, HEIGHT);
    glutCreateWindow("Cosine Curve");

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0, N, -1.5, 1.5, -1, 1);

    glClearColor(0.0, 0.0, 0.0, 0.0);
    glClear(GL_COLOR_BUFFER_BIT);

    glColor3f(1.0, 1.0, 1.0);
    glBegin(GL_LINE_STRIP);
    for (int i = 0; i < N; i++)
    {
        glVertex2f(i, d_output[i]);
    }
    glEnd();

    glFlush();

    cudaFree(d_output);
    delete[] h_output;

    glutMainLoop();

    return 0;
}