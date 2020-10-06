# interior_point_method_for_QP
For comprehensive implementation from Algorithm 16.4 (Predictor-Corrector Algorithm for QP) in Nocedal &amp; Wright (2006)

"interior_point.py" solves convex quadratic programming problems as follows:

  min_x x^T Gx + q^T x + const.
  s.t.  Ax≧b,

where ^T denotes transpose of a vector. Here, G is positive semidefinite and A is possibly column full rank matrix.
