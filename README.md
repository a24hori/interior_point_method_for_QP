# interior_point_method_for_QP
For comprehensive implementation of Algorithm 16.4 (Predictor-Corrector Algorithm for QP) in Nocedal &amp; Wright (2006)

"interior_point.py" solves the following convex quadratic programming problems:

  min_x x^T Gx + q^T x + const.
  s.t.  Ax≧b,

where ^T denotes transpose of a vector. Here, G is positive semidefinite and A is possibly column full rank matrix.
