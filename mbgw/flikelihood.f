! This Fortran implementation of the likelihood function, up to
! a normalizing constant, is much faster than its Python equivalent
! would be. Since the likelihood function is evaluated an enormous
! number of times over the course of the computations, it is worth
! optimizing heavily.
      SUBROUTINE binomial(d, n, f, nf, lp)
cf2py intent(hide) nf
cf2py intent(out) lp
      DOUBLE PRECISION d,n,f(nf),lp(nf)
      INTEGER i

      do i=1,nf
          lp(i) = d*f(i)-n*dlog(1.0D0+dexp(f(i)))
      end do

      RETURN
      END