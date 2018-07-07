// Copyright (C) 2015-2018, Lehigh University All Rights Reserved.
// This code is licensed under the terms of the Eclipse Public License (EPL).

#ifndef OsiIpoptTNLP_H
#define OsiIpoptTNLP_H

#include "IpTNLP.hpp"

class CoinPackedMatrix;

using namespace Ipopt;

class OsiIpoptTNLP : public TNLP {
  CoinPackedMatrix const * const matrix_;
  CoinPackedMatrix const * const rev_matrix_;
  double const * const rowlb_;
  double const * const rowub_;
  double const * const collb_;
  double const * const colub_;
  double const * const obj_;
  int const numCones_;
  int const * const coneSize_;
  // 1 for Lorentz cones
  // 2 for rotated Lorentz cones
  int const * const coneType_;
  int const * const * const coneMembers_;
  double * solution_;
public:
  OsiIpoptTNLP(CoinPackedMatrix const * const matrix,
                      CoinPackedMatrix const * const rev_matrix,
                      double const * const rowlb,
                      double const * const rowub,
                      double const * const collb,
                      double const * const colub,
                      double const * const obj,
                      int const numCones,
                      int const * const coneSize,
                      int const * const coneType,
                      int const * const * const coneMembers);
  virtual ~OsiIpoptTNLP();
  double const * solution() const { return solution_; }
  /// @name Virtual functions of Ipopt, inherited from TNLP
  //@{
  /** Method to return some info about the nlp */
  virtual bool get_nlp_info(Index& n, Index& m, Index& nnz_jac_g,
                            Index& nnz_h_lag, IndexStyleEnum& index_style);
  /** Method to return the bounds for my problem */
  virtual bool get_bounds_info(Index n, Number* x_l, Number* x_u,
                               Index m, Number* g_l, Number* g_u);
  /** Method to return the starting point for the algorithm */
  virtual bool get_starting_point(Index n, bool init_x, Number* x,
                                  bool init_z, Number* z_L, Number* z_U,
                                  Index m, bool init_lambda,
                                  Number* lambda);
  /** Method to return the objective value */
  virtual bool eval_f(Index n, const Number* x, bool new_x, Number& obj_value);
  /** Method to return the gradient of the objective */
  virtual bool eval_grad_f(Index n, const Number* x, bool new_x, Number* grad_f);
  /** Method to return the constraint residuals */
  virtual bool eval_g(Index n, const Number* x, bool new_x, Index m, Number* g);
  /** Method to return:
   *   1) The structure of the jacobian (if "values" is NULL)
   *   2) The values of the jacobian (if "values" is not NULL)
   */
  virtual bool eval_jac_g(Index n, const Number* x, bool new_x,
                          Index m, Index nele_jac, Index* iRow, Index *jCol,
                          Number* values);
  /** Method to return:
   *   1) The structure of the hessian of the lagrangian (if "values" is NULL)
   *   2) The values of the hessian of the lagrangian (if "values" is not NULL)
   */
  virtual bool eval_h(Index n, const Number* x, bool new_x,
                      Number obj_factor, Index m, const Number* lambda,
                      bool new_lambda, Index nele_hess, Index* iRow,
                      Index* jCol, Number* values);
  /** This method is called when the algorithm is complete so the TNLP can store/write the solution */
  virtual void finalize_solution(SolverReturn status,
                                 Index n, const Number* x, const Number* z_L, const Number* z_U,
                                 Index m, const Number* g, const Number* lambda,
                                 Number obj_value,
                                 const IpoptData* ip_data,
                                 IpoptCalculatedQuantities* ip_cq);

  //@}
private:
  OsiIpoptTNLP(OsiIpoptTNLP const &);
  OsiIpoptTNLP & operator=(OsiIpoptTNLP const &);
};

#endif
