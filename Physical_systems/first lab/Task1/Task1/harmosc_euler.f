      PROGRAM one_d_motion2

      IMPLICIT NONE

      INCLUDE 'globals.inc'

      character(len=32) :: ARG
      integer, parameter :: IN = 1, NCYCLES = 2
      integer :: N = 100, I
      real(db), allocatable :: T(:), X(:), V(:)
      real(db), allocatable :: XANAL(:), VANAL(:)
      real :: DT


!     Number of steps can be given as input argument
      IF (IARGC() > 0) THEN
        CALL GETARG(1,ARG)
        READ(ARG,'(I10)') N
      ENDIF
      ! cycles (of 2*pi)
      DT = NCYCLES*2.0*pi/N
      WRITE(*,*) '#INFO: nr. of steps: ',N
      WRITE(*,*) '#INFO: nr. of cycles: ',NCYCLES
      WRITE(*,*) '#INFO: time step: ',DT
      WRITE(*,*) '#       time            x              v     &
&       x-analytical     v-analytical' 

!     Allocate arrays
      ALLOCATE(T(N+1),X(N+1),V(N+1))
      ALLOCATE(XANAL(N+1),VANAL(N+1))

!     Initial values
      X(1) = 0.0
      V(1) = 1.0
      T(1) = 0.0
      XANAL(1) = X(1)
      VANAL(1) = V(1)

      DO I = 1,N
         T(I+1) = I*DT

!     Analytic solution
         XANAL(I+1) = SIN(T(I+1))
         VANAL(I+1) = COS(T(I+1))

!     Euler (predictor) for position and velocity
         X(I+1) = X(I) + V(I)*DT
         V(I+1) = V(I) - X(I)*DT

      END DO

      WRITE (*,'(5F16.8)') (T(I),X(I),V(I),XANAL(I),VANAL(I),I=1,N+1,IN)

      DEALLOCATE(X,V,T)

      END
