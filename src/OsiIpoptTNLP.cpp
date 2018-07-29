
#include <cmath>
#include <numeric>

#include "OsiIpoptTNLP.hpp"

#include "CoinPackedMatrix.hpp"

OsiIpoptTNLP::OsiIpoptTNLP(CoinPackedMatrix const * const matrix,
                           CoinPackedMatrix const * const rev_matrix,
                           double const * const rowlb,
                           double const * const rowub,
                           double const * const collb,
                           double const * const colub,
                           double const * const obj,
                           int const numCones,
                           int const * const coneSize,
                           int const * const coneType,
                           int const * const * const coneMembers):
  matrix_(matrix), rev_matrix_(rev_matrix), rowlb_(rowlb),
  rowub_(rowub), collb_(collb), colub_(colub), obj_(obj),
  numCones_(numCones), coneSize_(coneSize), coneType_(coneType),
  coneMembers_(coneMembers), solution_(NULL) {
}

OsiIpoptTNLP::~OsiIpoptTNLP() {
  if (solution_) {
    delete[] solution_;
  }
}


bool OsiIpoptTNLP::get_nlp_info(Index& n, Index& m,
                                Index& nnz_jac_g,
                                Index& nnz_h_lag,
                                IndexStyleEnum& index_style) {
  // The problem described in CutProblem.hpp has size_ variables
  n = matrix_->getNumCols();
  //  nonzeros in the jacobian
  m = matrix_->getNumRows() + numCones_;
  nnz_jac_g = matrix_->getNumElements();
  for (int i=0; i<numCones_; ++i) {
    nnz_jac_g += coneSize_[i];
  }
  // nonzeros in the hessian of the lagrangian
  nnz_h_lag = 0;
  for (int i=0; i<numCones_; ++i) {
    nnz_h_lag += coneSize_[i];
  }
  // We use C index style for row/col entries
  index_style = TNLP::C_STYLE;
  return true;
}

/** Method to return the bounds for my problem */
bool OsiIpoptTNLP::get_bounds_info(Index n, Number* x_l,
                                   Number* x_u,
                                   Index m, Number* g_l,
                                   Number* g_u) {
  // here, the n and m we gave IPOPT in get_nlp_info are passed back to us.
  // If desired, we could assert to make sure they are what we think they are.
  assert(n == matrix_->getNumCols());
  assert(m == (matrix_->getNumRows() + numCones_));
  // bounds on variables
  std::copy(collb_, collb_+n, x_l);
  std::copy(colub_, colub_+n, x_u);
  // set lower bound to 0 for leading variables of the cones.
  for (int i=0; i<numCones_; ++i) {
    if (coneType_[i]==1) {
      if (collb_[coneMembers_[i][0]]<0.0) {
        x_l[coneMembers_[i][0]] = 0.0;
      }
    }
    else if (coneType_[i]==2) {
      if (collb_[coneMembers_[i][0]]<0.0) {
        x_l[coneMembers_[i][0]] = 0.0;
      }
      if (collb_[coneMembers_[i][1]]<0.0) {
        x_l[coneMembers_[i][1]] = 0.0;
      }
    }
  }
  // bound on constraints
  int num_rows = matrix_->getNumRows();
  std::copy(rowlb_, rowlb_+num_rows, g_l);
  std::fill(g_l+num_rows, g_l+m, 0.0);
  std::copy(rowub_, rowub_+num_rows, g_u);
  std::fill(g_u+num_rows, g_u+m, 2e19);
  return true;
}

/** Method to return the starting point for the algorithm */
bool OsiIpoptTNLP::get_starting_point(Index n, bool init_x,
                                      Number* x,
                                      bool init_z, Number* z_L,
                                      Number* z_U,
                                      Index m,
                                      bool init_lambda,
                                      Number* lambda) {
  if (solution_==NULL) {
    assert(init_x == true);
    assert(init_z == false);
    assert(init_lambda == false);
    std::fill(x, x+n, 1.0);
    // set cone leading variables
    for (int i=0; i<numCones_; ++i) {
      if (coneType_[i]==1) {
        x[coneMembers_[i][0]] = sqrt(double(coneSize_[i]));
      }
      else if (coneType_[i]==2) {
        double val = sqrt(double(coneSize_[i])/2.0);
        x[coneMembers_[i][0]] = val;
        x[coneMembers_[i][1]] = val;
      }
    }
  }
  else {
    std::copy(solution_, solution_+n, x);
  }
  return true;
}

/** Method to return the objective value */
bool OsiIpoptTNLP::eval_f(Index n, const Number* x,
                          bool new_x, Number& obj_value) {
  // compute c^Tx
  obj_value = std::inner_product(obj_, obj_+n, x, 0.0);
  return true;
}

/** Method to return the gradient of the objective */
bool OsiIpoptTNLP::eval_grad_f(Index n, const Number* x,
                               bool new_x, Number* grad_f) {
  std::copy(obj_, obj_+n, grad_f);
  return true;
}

/** Method to return the constraint residuals */
bool OsiIpoptTNLP::eval_g(Index n, const Number* x,
                          bool new_x, Index m, Number* g) {
  // first rows are Ax
  int num_rows = matrix_->getNumRows();
  double * Ax = new double[num_rows];
  matrix_->times(x, Ax);
  std::copy(Ax, Ax+num_rows, g);
  for (int i=0; i<numCones_; ++i) {
    int const * mem = coneMembers_[i];
    double term1;
    int start;
    if (coneType_[i]==1) {
      term1 = x[mem[0]]*x[mem[0]];
      start = 1;
    }
    else if (coneType_[i]==2) {
      term1 = 2.0*x[mem[0]]*x[mem[1]];
      start = 2;
    }
    double term2 = 0.0;
    for (int j=start; j<coneSize_[i]; ++j) {
      term2 += x[mem[j]]*x[mem[j]];
    }
    g[i+num_rows] = term1 - term2;
  }
  delete[] Ax;
  return true;
}

/** Method to return:
 *   1) The structure of the jacobian (if "values" is NULL)
 *   2) The values of the jacobian (if "values" is not NULL)
 */
bool OsiIpoptTNLP::eval_jac_g(Index n, const Number* x, bool new_x,
                              Index m, Index nele_jac,
                              Index* iRow, Index *jCol,
                              Number* values) {
  int const * indices = matrix_->getIndices();
  double const * elements = matrix_->getElements();
  int num_rows = matrix_->getNumRows();
  int num_elem = 0;
  if (values==NULL) {
    // insert structure of the A matrix first
    // matrix_ is column ordered
    for (int j=0; j<n; ++j) {
      int first = matrix_->getVectorFirst(j);
      int last = matrix_->getVectorLast(j);
      std::fill(jCol+num_elem, jCol+num_elem+last-first, j);
      std::copy(indices+first, indices+last, iRow+num_elem);
      num_elem += (last-first);
    }
    // end of matrix A, now insert for conic constraints
    for (int i=0; i<numCones_; ++i) {
      std::fill(iRow+num_elem, iRow+num_elem+coneSize_[i], i+num_rows);
      std::copy(coneMembers_[i], coneMembers_[i]+coneSize_[i], jCol+num_elem);
      num_elem += coneSize_[i];
    }
  }
  else {
    // insert values of the A matrix first
    // matrix_ is column ordered
    std::copy(elements, elements+matrix_->getNumElements(), values);
    num_elem = matrix_->getNumElements();
    int start;
    // end of matrix A, now insert for conic constraints
    for (int i=0; i<numCones_; ++i) {
      if (coneType_[i]==1) {
        values[num_elem] = 2.0*x[coneMembers_[i][0]];
        num_elem++;
        start = 1;
      }
      else if (coneType_[i]==2) {
        values[num_elem] = 2.0*x[coneMembers_[i][1]];
        num_elem++;
        values[num_elem] = 2.0*x[coneMembers_[i][0]];
        num_elem++;
        start = 2;
      }
      for (int j=start; j<coneSize_[i]; ++j) {
        values[num_elem] = -2.0*x[coneMembers_[i][j]];
        num_elem++;
      }
    }
  }
  return true;
}

/** Method to return:
 *   1) The structure of the hessian of the lagrangian (if "values" is NULL)
 *   2) The values of the hessian of the lagrangian (if "values" is not NULL)
 */
bool OsiIpoptTNLP::eval_h(Index n, const Number* x, bool new_x,
                          Number obj_factor, Index m,
                          const Number* lambda,
                          bool new_lambda,
                          Index nele_hess, Index* iRow,
                          Index* jCol, Number* values) {
  int num_elem = 0;
  int num_rows = matrix_->getNumRows();
  if (values==NULL) {
    // Hessian of the linear part is 0.
    // Relevant part corresponds to conic constraints.
    for (int i=0; i<numCones_; ++i) {
      std::copy(coneMembers_[i], coneMembers_[i]+coneSize_[i], iRow+num_elem);
      std::copy(coneMembers_[i], coneMembers_[i]+coneSize_[i], jCol+num_elem);
      num_elem += coneSize_[i];
    }
  }
  else {
    // Hessian of the linear part is 0.
    // Relevant part corresponds to conic constraints.
    for (int i=0; i<numCones_; ++i) {
      double dual = lambda[num_rows+i];
      // fill -2dual for all cone members
      std::fill(values+num_elem, values+num_elem+coneSize_[i], -2.0*dual);
      // change the value for the leading variable
      if (coneType_[i]==1) {
        values[num_elem] = 2.0*dual;
      }
      else if (coneType_[i]==2) {
        values[num_elem] = 2.0*dual;
        values[num_elem+1] = 2.0*dual;
      }
      num_elem += coneSize_[i];
    }
  }
  return true;
}

/** This method is called when the algorithm is complete so the
TNLP can store/write the solution */
void OsiIpoptTNLP::finalize_solution(SolverReturn status,
                                     Index n, const Number* x,
                                     const Number* z_L,
                                     const Number* z_U,
                                     Index m, const Number* g,
                                     const Number* lambda,
                                     Number obj_value,
                                     const IpoptData* ip_data,
                                     IpoptCalculatedQuantities* ip_cq) {
  // here is where we would store the solution to variables, or write to a file, etc
  // so we could use the solution.
  if (solution_==NULL) {
    solution_ = new double[n];
  }
  std::copy(x, x+n, solution_);
}
