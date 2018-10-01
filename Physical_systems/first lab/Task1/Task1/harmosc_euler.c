#include<stdio.h>
#include<stdlib.h>
#include<math.h>

/* a naive port from harmosc_euler.f
 * Wei Chen, 2013 */

int main(int argc, char *argv[])
{
    int in = 1, ncycles = 2, n = 100, i;
    if (argc == 2) n = atoi(argv[1]);
    double t[n+1], x[n+1], v[n+1], xanaly[n+1], vanaly[n+1];
    double dt, pi;
    /* t[]: time
       x[]: displacement
       v[]: velocity
       xanaly, vanaly: analytical solutions*/

    pi = 4.0*atan(1.0);
    dt = ncycles*2.0*pi/n;   

    printf("#INFO: number of steps:  %d\n", n);
    printf("#INFO: number of cycles: %d\n", ncycles);
    printf("#INFO: time step:        %10.6f\n", dt);
    printf("#       time                x               v           x-analytical      v-analytical\n");

    x[0] = 0.0; v[0] = 1.0; t[0] = 0.0;
    xanaly[0] = x[0]; vanaly[0] = v[0];

    for (i = 1; i < n+1; i++) 
        {
            t[i] = i*dt;
            xanaly[i] = sin(t[i]);
            vanaly[i] = cos(t[i]);
            x[i] = x[i-1] + v[i-1]*dt;
            v[i] = v[i-1] - x[i-1]*dt;
        }

    for (i = 0; i < n+1; i++)
        { 
            printf("%16.8f %16.8f %16.8f %16.8f %16.8f\n", t[i], x[i], v[i], xanaly[i], vanaly[i]);
        }
        
}
