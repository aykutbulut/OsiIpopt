// Copyright (C) 2015, Lehigh University All Rights Reserved.
// This code is licensed under the terms of the Eclipse Public License (EPL).

#include <exception>
#include <iostream>
#include <cmath>
#include <numeric>

#include <IpSolveStatistics.hpp>

#include "OsiIpoptSolverInterface.hpp"

// TODO(aykut):
// 1. Implement initialSolve
// 2. Implement loadProblem, done.
// 2. Implement addConicConstraint, done.
// 3. implement setColName
// 4. implement setRowName
// 5. implement setInteger

// the NLP problem we have is as follows,
// min c^Tx
// s.t. lb <= Ax <= ub
//      x^1 in L^1, ie. x^1_1 ^2 - x^{1T}x^1 >= 0
//      x^2 in L^2
//      .
//      .
//      x^k in L^k
//
// Then following are the values needed for Ipopt at point x,
// f(x) = c^Tx
// grad_f(x) = c
// h(x) = 0
// g(x) = [Ax                                                 ]
//        [-x^{1T} Jx^{1} 0                     0             ]
//        [ 0             -x^{2T} Jx^{2} 0      0             ]
//        [                                     -x^{kT}Jx^{k} ]
// gl = [lb; 0]
// gu = [ub; inf]
// J(x) = [A                                                                                   ;
//         2x^1_1 -2x^1_2 -2x^1_3 ... -2x^1_{n_1} 0       0      ...                          0;
//         0       0       0 ...       0          2x^2_1 -2x^2_2 ... -2x^2_{n_2} 0 ... 0 0 ...0;
//         .
//         .
//         0  0  0 ...  0 0  0 ...  0 0 ... 0                                            2x^k]
// H(x) = [-2u_1 J   0       0   ...  0
//          0       -2u_2 J  0   ...  0
//          0        0      -2J  ...  0
//          .   .
//          .   .
//          0   0      ...           -2u_k J]
// where J is [-1 0; 0 I] with the right size, and u_i are the lagrange multipliers.
//


#define CONE_CHUNK 100

//#############################################################################
// Solve methods
//#############################################################################
void OsiIpoptSolverInterface::initialSolve() {
  app_->Options()->SetIntegerValue("print_level", 0);
  Ipopt::ApplicationReturnStatus status;
  TNLP * tnlp_p = dynamic_cast<TNLP*>(this);
  status_ = app_->OptimizeTNLP(tnlp_p);
}

void OsiIpoptSolverInterface::resolve() {
  app_->Options()->SetIntegerValue("print_level", 0);
  initialSolve();
}

void OsiIpoptSolverInterface::branchAndBound() {
  throw IpoptException("Not implemented yet!", __FILE__, __LINE__, std::string("OsiIpopt exception"));
}

//#############################################################################
// Querrying solution status
//#############################################################################
bool OsiIpoptSolverInterface::isAbandoned() const {
  if (status_==Solve_Succeeded or status_==Solved_To_Acceptable_Level
      or status_==Infeasible_Problem_Detected)
    return false;
  else
    return true;
}

bool OsiIpoptSolverInterface::isProvenOptimal() const {
  if (status_==Solve_Succeeded or status_==Solved_To_Acceptable_Level)
    return true;
  else
    return false;
}

bool OsiIpoptSolverInterface::isProvenPrimalInfeasible() const {
  if (status_==Infeasible_Problem_Detected)
    return true;
  else
    return false;
}

bool OsiIpoptSolverInterface::isProvenDualInfeasible() const {
  if (status_==Infeasible_Problem_Detected)
    return true;
  else
    return false;
}

bool OsiIpoptSolverInterface::isPrimalObjectiveLimitReached() const {
  if (status_==Diverging_Iterates)
    return true;
  else
    return false;
}

bool OsiIpoptSolverInterface::isDualObjectiveLimitReached() const {
  if (status_==Diverging_Iterates)
    return true;
  else
    return false;
}

bool OsiIpoptSolverInterface::isIterationLimitReached() const {
  if (status_==Maximum_Iterations_Exceeded)
    return true;
  else
    return false;
}

//#############################################################################
// Problem Query methods
//#############################################################################
/// Get the number of columns
int OsiIpoptSolverInterface::getNumCols() const {
  int n = matrix_->getNumCols();
  return n;
}

/// Get the number of rows
int OsiIpoptSolverInterface::getNumRows() const {
  int m = matrix_->getNumRows();
  return m;
}

/// Get the number of nonzero elements
int OsiIpoptSolverInterface::getNumElements() const {
  throw IpoptException("Not implemented yet!", __FILE__, __LINE__, std::string("OsiIpopt exception"));
  return 0;
}

/// Get a pointer to an array[getNumCols()] of column lower bounds
const double * OsiIpoptSolverInterface::getColLower() const {
  return collb_;
}

/// Get a pointer to an array[getNumCols()] of column upper bounds
const double * OsiIpoptSolverInterface::getColUpper() const {
  return colub_;
}

/*! \brief Get a pointer to an array[getNumRows()] of row constraint senses.

    <ul>
    <li>'L': <= constraint
    <li>'E': =  constraint
    <li>'G': >= constraint
    <li>'R': ranged constraint
    <li>'N': free constraint
    </ul>
*/
const char * OsiIpoptSolverInterface::getRowSense() const {
  throw IpoptException("Not implemented yet!", __FILE__, __LINE__, std::string("OsiIpopt exception"));
  return 0;
}

/*! \brief Get a pointer to an array[getNumRows()] of row right-hand sides

    <ul>
    <li> if getRowSense()[i] == 'L' then
    getRightHandSide()[i] == getRowUpper()[i]
    <li> if getRowSense()[i] == 'G' then
    getRightHandSide()[i] == getRowLower()[i]
    <li> if getRowSense()[i] == 'R' then
    getRightHandSide()[i] == getRowUpper()[i]
    <li> if getRowSense()[i] == 'N' then
    getRightHandSide()[i] == 0.0
    </ul>
*/
const double * OsiIpoptSolverInterface::getRightHandSide() const {
  throw IpoptException("Not implemented yet!", __FILE__, __LINE__, std::string("OsiIpopt exception"));
  return 0;
}

/*! \brief Get a pointer to an array[getNumRows()] of row ranges.

    <ul>
    <li> if getRowSense()[i] == 'R' then
    getRowRange()[i] == getRowUpper()[i] - getRowLower()[i]
    <li> if getRowSense()[i] != 'R' then
    getRowRange()[i] is 0.0
    </ul>
*/
const double * OsiIpoptSolverInterface::getRowRange() const {
  throw IpoptException("Not implemented yet!", __FILE__, __LINE__, std::string("OsiIpopt exception"));
  return 0;
}

/// Get a pointer to an array[getNumRows()] of row lower bounds
const double * OsiIpoptSolverInterface::getRowLower() const {
  return rowlb_;
}

/// Get a pointer to an array[getNumRows()] of row upper bounds
const double * OsiIpoptSolverInterface::getRowUpper() const {
  return rowub_;
}

/*! \brief Get a pointer to an array[getNumCols()] of objective
    function coefficients.
*/
const double * OsiIpoptSolverInterface::getObjCoefficients() const {
  if (obj_==NULL) {
    throw IpoptException("Objctive coef not allocated!", __FILE__,
                         __LINE__, std::string("OsiIpopt exception"));
  }
  return obj_;
}

/*! \brief Get the objective function sense

    -  1 for minimisation (default)
    - -1 for maximisation
*/
double OsiIpoptSolverInterface::getObjSense() const {
  return 1.0;
}

/// Return true if the variable is continuous
bool OsiIpoptSolverInterface::isContinuous(int colIndex) const {
  throw IpoptException("Not implemented yet!", __FILE__, __LINE__, std::string("OsiIpopt exception"));
  return 0;
}

/// Return true if the variable is binary
bool OsiIpoptSolverInterface::isBinary(int colIndex) const {
  throw IpoptException("Not implemented yet!", __FILE__, __LINE__, std::string("OsiIpopt exception"));
  return 0;
}

/*! \brief Return true if the variable is integer.

  This method returns true if the variable is binary or general integer.
*/
bool OsiIpoptSolverInterface::isInteger(int colIndex) const {
  bool res = false;
  if (varType_[colIndex] == INTEGER) {
    res = true;
  }
  return res;
}

/// Get a pointer to a row-wise copy of the matrix
const CoinPackedMatrix * OsiIpoptSolverInterface::getMatrixByRow() const {
  if (matrix_->isColOrdered()) {
    return rev_matrix_;
  }
  return matrix_;
}

/// Get a pointer to a column-wise copy of the matrix
const CoinPackedMatrix * OsiIpoptSolverInterface::getMatrixByCol() const {
  if (matrix_->isColOrdered()) {
    return matrix_;
  }
  return rev_matrix_;
}

/// Get the solver's value for infinity
double OsiIpoptSolverInterface::getInfinity() const {
  throw IpoptException("Not implemented yet!", __FILE__, __LINE__, std::string("OsiIpopt exception"));
  return 0.0;
}

//#############################################################################
// Solution Query methods
//#############################################################################
/// Get a pointer to an array[getNumCols()] of primal variable values
const double * OsiIpoptSolverInterface::getColSolution() const {
  return solution_;
}

/// Get pointer to array[getNumRows()] of dual variable values
const double * OsiIpoptSolverInterface::getRowPrice() const {
  throw IpoptException("Not implemented yet!", __FILE__, __LINE__, std::string("OsiIpopt exception"));
  return 0;
}

/// Get a pointer to an array[getNumCols()] of reduced costs
const double * OsiIpoptSolverInterface::getReducedCost() const {
  throw IpoptException("Not implemented yet!", __FILE__, __LINE__, std::string("OsiIpopt exception"));
  return 0;
}

/** Get a pointer to array[getNumRows()] of row activity levels.
    The row activity for a row is the left-hand side evaluated at the
    current solution.
*/
const double * OsiIpoptSolverInterface::getRowActivity() const {
  throw IpoptException("Not implemented yet!", __FILE__, __LINE__, std::string("OsiIpopt exception"));
  return 0;
}

/// Get the objective function value.
double OsiIpoptSolverInterface::getObjValue() const {
  int n = matrix_->getNumCols();
  double val = 1e+30;
  if (solution_) {
     val = std::inner_product(obj_, obj_+n, solution_, 0.0);
  }
  return val;
}

/** Get the number of iterations it took to solve the problem (whatever
    `iteration' means to the solver).
*/
int OsiIpoptSolverInterface::getIterationCount() const {

  return app_->Statistics()->IterationCount();
}

/** Get as many dual rays as the solver can provide. In case of proven
    primal infeasibility there should (with high probability) be at least
    one.

    The first getNumRows() ray components will always be associated with
    the row duals (as returned by getRowPrice()). If \c fullRay is true,
    the final getNumCols() entries will correspond to the ray components
    associated with the nonbasic variables. If the full ray is requested
    and the method cannot provide it, it will throw an exception.

    \note
    Implementors of solver interfaces note that the double pointers in
    the vector should point to arrays of length getNumRows() (fullRay =
    false) or (getNumRows()+getNumCols()) (fullRay = true) and they should
    be allocated with new[].

    \note
    Clients of solver interfaces note that it is the client's
    responsibility to free the double pointers in the vector using
    delete[]. Clients are reminded that a problem can be dual and primal
    infeasible.
*/
std::vector<double*> OsiIpoptSolverInterface::getDualRays(
                                          int maxNumRays,
                                          bool fullRay) const {
  throw IpoptException("Not implemented yet!", __FILE__, __LINE__, std::string("OsiIpopt exception"));
  std::vector<double*> rays;
  return rays;
}

/** Get as many primal rays as the solver can provide. In case of proven
    dual infeasibility there should (with high probability) be at least
    one.

    \note
    Implementors of solver interfaces note that the double pointers in
    the vector should point to arrays of length getNumCols() and they
    should be allocated with new[].

    \note
    Clients of solver interfaces note that it is the client's
    responsibility to free the double pointers in the vector using
    delete[]. Clients are reminded that a problem can be dual and primal
    infeasible.
*/
std::vector<double*> OsiIpoptSolverInterface::getPrimalRays(
                                            int maxNumRays) const {
  throw IpoptException("Not implemented yet!", __FILE__, __LINE__, std::string("OsiIpopt exception"));
  std::vector<double*> rays;
  return rays;
}

//#############################################################################
// Methods to modify objective, bounds and solution
//#############################################################################
/** Set an objective function coefficient */
void OsiIpoptSolverInterface::setObjCoeff( int elementIndex,
                                           double elementValue ) {
  throw IpoptException("Not implemented yet!", __FILE__, __LINE__, std::string("OsiIpopt exception"));
}

/** Set the objective function sense.
    Use 1 for minimisation (default), -1 for maximisation.
    \note
    Implementors note that objective function sense is a parameter of
    the OSI, not a property of the problem. Objective sense can be
    set prior to problem load and should not be affected by loading a
    new problem.
*/
void OsiIpoptSolverInterface::setObjSense(double s) {
  if (s==1.0) {
    // all is fine, no need to do anything.
  }
  else if (s==-1) {
    throw IpoptException("Not implemented yet!", __FILE__, __LINE__, std::string("OsiIpopt exception"));
  }
  else {
    throw IpoptException("Not implemented yet!", __FILE__, __LINE__, std::string("OsiIpopt exception"));
  }
}

/** Set a single column lower bound.
    Use -getInfinity() for -infinity. */
void OsiIpoptSolverInterface::setColLower( int elementIndex,
                                           double elementValue ) {
  collb_[elementIndex] = elementValue;
}

/** Set a single column upper bound.
    Use getInfinity() for infinity. */
void OsiIpoptSolverInterface::setColUpper( int elementIndex,
                                           double elementValue ) {
  colub_[elementIndex] = elementValue;
}

/** Set a single row lower bound.
    Use -getInfinity() for -infinity. */
void OsiIpoptSolverInterface::setRowLower( int elementIndex,
                                           double elementValue ) {
  rowlb_[elementIndex] = elementValue;
}

/** Set a single row upper bound.
    Use getInfinity() for infinity. */
void OsiIpoptSolverInterface::setRowUpper( int elementIndex,
                                           double elementValue ) {
  rowub_[elementIndex] = elementValue;
}

/** Set the type of a single row */
void OsiIpoptSolverInterface::setRowType(int index, char sense,
                                         double rightHandSide,
                double range) {
  throw IpoptException("Not implemented yet!", __FILE__, __LINE__, std::string("OsiIpopt exception"));
}

/// Set a hint parameter
bool OsiIpoptSolverInterface::setHintParam(OsiHintParam key,
                      bool yesNo,
                      OsiHintStrength strength,
                      void * otherInformation) {
  if (key==OsiDoReducePrint) {
    if (yesNo) {
      app_->Options()->SetIntegerValue("print_level", 0);
    }
    else {
      app_->Options()->SetIntegerValue("print_level", 5);
    }
  }
  else {
    throw IpoptException("Not implemented yet!", __FILE__, __LINE__,
                         std::string("OsiIpopt exception"));
  }
}

/** Set the primal solution variable values

    colsol[getNumCols()] is an array of values for the primal variables.
    These values are copied to memory owned by the solver interface
    object or the solver.  They will be returned as the result of
    getColSolution() until changed by another call to setColSolution() or
    by a call to any solver routine.  Whether the solver makes use of the
    solution in any way is solver-dependent.
*/
void OsiIpoptSolverInterface::setColSolution(const double *colsol)  {
  // do nothing or now
}

/** Set dual solution variable values

    rowprice[getNumRows()] is an array of values for the dual variables.
    These values are copied to memory owned by the solver interface
    object or the solver.  They will be returned as the result of
    getRowPrice() until changed by another call to setRowPrice() or by a
    call to any solver routine.  Whether the solver makes use of the
    solution in any way is solver-dependent.
*/
void OsiIpoptSolverInterface::setRowPrice(const double * rowprice) {
  throw IpoptException("Not implemented yet!", __FILE__, __LINE__, std::string("OsiIpopt exception"));
}

//#############################################################################
// Methods to set variable type
//#############################################################################
/** Set the index-th variable to be a continuous variable */
void OsiIpoptSolverInterface::setContinuous(int index) {
  throw IpoptException("Not implemented yet!", __FILE__, __LINE__, std::string("OsiIpopt exception"));
}

/** Set the index-th variable to be an integer variable */
void OsiIpoptSolverInterface::setInteger(int index) {
  varType_[index] = INTEGER;
}

//#############################################################################
// Methods to modify the constraint system.
//#############################################################################
/** Add a column (primal variable) to the problem. */
void OsiIpoptSolverInterface::addCol(const CoinPackedVectorBase& vec,
            const double collb, const double colub,
            const double obj) {
  throw IpoptException("Not implemented yet!", __FILE__, __LINE__, std::string("OsiIpopt exception"));
}

/** \brief Remove a set of columns (primal variables) from the
    problem.

    The solver interface for a basis-oriented solver will maintain valid
    warm start information if all deleted variables are nonbasic.
*/
void OsiIpoptSolverInterface::deleteCols(const int num,
                                         const int * colIndices) {
  throw IpoptException("Not implemented yet!", __FILE__, __LINE__, std::string("OsiIpopt exception"));
}

/*! \brief Add a row (constraint) to the problem. */
void OsiIpoptSolverInterface::addRow(const CoinPackedVectorBase& vec,
            const double rowlb, const double rowub) {
  throw IpoptException("Not implemented yet!", __FILE__, __LINE__, std::string("OsiIpopt exception"));
}

/*! \brief Add a row (constraint) to the problem. */
void OsiIpoptSolverInterface::addRow(const CoinPackedVectorBase& vec,
                                     const char rowsen, const double rowrhs,
                                     const double rowrng) {
  throw IpoptException("Not implemented yet!", __FILE__, __LINE__, std::string("OsiIpopt exception"));
}

/** \brief Delete a set of rows (constraints) from the problem.

    The solver interface for a basis-oriented solver will maintain valid
    warm start information if all deleted rows are loose.
*/
void OsiIpoptSolverInterface::deleteRows(const int num,
                                         const int * rowIndices) {
  throw IpoptException("Not implemented yet!", __FILE__, __LINE__, std::string("OsiIpopt exception"));
}

//#############################################################################
// Methods for problem input and output
//#############################################################################
/*! \brief Load in a problem by copying the arguments. The constraints on
  the rows are given by lower and upper bounds.

  If a pointer is 0 then the following values are the default:
  <ul>
  <li> <code>colub</code>: all columns have upper bound infinity
  <li> <code>collb</code>: all columns have lower bound 0
  <li> <code>rowub</code>: all rows have upper bound infinity
  <li> <code>rowlb</code>: all rows have lower bound -infinity
  <li> <code>obj</code>: all variables have 0 objective coefficient
  </ul>

  Note that the default values for rowub and rowlb produce the
  constraint -infty <= ax <= infty. This is probably not what you want.
*/
void OsiIpoptSolverInterface::loadProblem (
                       const CoinPackedMatrix& matrix,
                       const double* collb, const double* colub,
                       const double* obj,
                       const double* rowlb, const double* rowub) {
  if (matrix_)
    delete matrix_;
  if (rev_matrix_)
    delete rev_matrix_;
  if (matrix.isColOrdered()) {
    matrix_ = new CoinPackedMatrix(matrix);
    rev_matrix_ = new CoinPackedMatrix();
    rev_matrix_->reverseOrderedCopyOf(matrix);
  }
  else {
    rev_matrix_ = new CoinPackedMatrix(matrix);
    matrix_ = new CoinPackedMatrix();
    matrix_->reverseOrderedCopyOf(matrix);
  }
  int m = matrix_->getNumRows();
  int n = matrix_->getNumCols();
  if (collb_)
    delete[] collb_;
  collb_ = new double[n];
  std::copy(collb, collb+n, collb_);
  if (colub_)
    delete[] colub_;
  colub_ = new double[n];
  std::copy(colub, colub+n, colub_);
  if (rowlb_)
    delete[] rowlb_;
  rowlb_ = new double[m];
  std::copy(rowlb, rowlb+m, rowlb_);
  if (rowub_)
    delete[] rowub_;
  rowub_ = new double[m];
  std::copy(rowub, rowub+m, rowub_);
  if (obj_)
    delete[] obj_;
  obj_ = new double[n];
  std::copy(obj, obj+n, obj_);
  varType_ = new VarType[n];
  // todo(aykut) set all variables to continuous for now
  std::fill(varType_, varType_+n, CONTINUOUS);
}

/*! \brief Load in a problem by assuming ownership of the arguments.
  The constraints on the rows are given by lower and upper bounds.

  For default argument values see the matching loadProblem method.

  \warning
  The arguments passed to this method will be freed using the
  C++ <code>delete</code> and <code>delete[]</code> functions.
*/
void OsiIpoptSolverInterface::assignProblem (
                            CoinPackedMatrix*& matrix,
                            double*& collb, double*& colub, double*& obj,
                            double*& rowlb, double*& rowub) {
  throw IpoptException("Not implemented yet!", __FILE__, __LINE__, std::string("OsiIpopt exception"));
}

/*! \brief Load in a problem by copying the arguments.
  The constraints on the rows are given by sense/rhs/range triplets.

  If a pointer is 0 then the following values are the default:
  <ul>
  <li> <code>colub</code>: all columns have upper bound infinity
  <li> <code>collb</code>: all columns have lower bound 0
  <li> <code>obj</code>: all variables have 0 objective coefficient
  <li> <code>rowsen</code>: all rows are >=
  <li> <code>rowrhs</code>: all right hand sides are 0
  <li> <code>rowrng</code>: 0 for the ranged rows
  </ul>

  Note that the default values for rowsen, rowrhs, and rowrng produce the
  constraint ax >= 0.
*/
void OsiIpoptSolverInterface::loadProblem (
                          const CoinPackedMatrix& matrix,
                          const double* collb, const double* colub,
                          const double* obj,
                          const char* rowsen, const double* rowrhs,
                          const double* rowrng) {
  throw IpoptException("Not implemented yet!", __FILE__, __LINE__, std::string("OsiIpopt exception"));
}

/*! \brief Load in a problem by assuming ownership of the arguments.
  The constraints on the rows are given by sense/rhs/range triplets.

  For default argument values see the matching loadProblem method.

  \warning
  The arguments passed to this method will be freed using the
  C++ <code>delete</code> and <code>delete[]</code> functions.
*/
void OsiIpoptSolverInterface::assignProblem (
                               CoinPackedMatrix*& matrix,
                               double*& collb, double*& colub, double*& obj,
                               char*& rowsen, double*& rowrhs,
                               double*& rowrng) {
  throw IpoptException("Not implemented yet!", __FILE__, __LINE__, std::string("OsiIpopt exception"));
}

/*! \brief Load in a problem by copying the arguments. The constraint
  matrix is is specified with standard column-major
  column starts / row indices / coefficients vectors.
  The constraints on the rows are given by lower and upper bounds.

  The matrix vectors must be gap-free. Note that <code>start</code> must
  have <code>numcols+1</code> entries so that the length of the last column
  can be calculated as <code>start[numcols]-start[numcols-1]</code>.

  See the previous loadProblem method using rowlb and rowub for default
  argument values.
*/
void OsiIpoptSolverInterface::loadProblem (const int numcols,
                                           const int numrows,
                                           const CoinBigIndex * start,
                                           const int* index,
                                           const double* value,
                                           const double* collb,
                                           const double* colub,
                                           const double* obj,
                                           const double* rowlb,
                                           const double* rowub) {
  throw IpoptException("Not implemented yet!", __FILE__, __LINE__, std::string("OsiIpopt exception"));
}

/*! \brief Load in a problem by copying the arguments. The constraint
  matrix is is specified with standard column-major
  column starts / row indices / coefficients vectors.
  The constraints on the rows are given by sense/rhs/range triplets.

  The matrix vectors must be gap-free. Note that <code>start</code> must
  have <code>numcols+1</code> entries so that the length of the last column
  can be calculated as <code>start[numcols]-start[numcols-1]</code>.

  See the previous loadProblem method using sense/rhs/range for default
  argument values.
*/
void OsiIpoptSolverInterface::loadProblem (const int numcols,
                                           const int numrows,
                                           const CoinBigIndex * start,
                                           const int* index,
                                           const double* value,
                                           const double* collb,
                                           const double* colub,
                                           const double* obj,
                                           const char* rowsen,
                                           const double* rowrhs,
                                           const double* rowrng) {
  throw IpoptException("Not implemented yet!", __FILE__, __LINE__, std::string("OsiIpopt exception"));
}

/*! \brief Write the problem in MPS format to the specified file.

  If objSense is non-zero, a value of -1.0 causes the problem to be
  written with a maximization objective; +1.0 forces a minimization
  objective. If objSense is zero, the choice is left to the implementation.
*/
void OsiIpoptSolverInterface::writeMps (const char * filename,
                                        const char * extension,
                                        double objSense) const {
  throw IpoptException("Not implemented yet!", __FILE__, __LINE__, std::string("OsiIpopt exception"));
}

///Protected methods of OsiSolverInterface
/** Apply a row cut (append to the constraint matrix). */
void OsiIpoptSolverInterface::applyRowCut( const OsiRowCut & rc ) {
  throw IpoptException("Not implemented yet!", __FILE__, __LINE__, std::string("OsiIpopt exception"));
}

/** Apply a column cut (adjust the bounds of one or more variables). */
void OsiIpoptSolverInterface::applyColCut( const OsiColCut & cc ) {
  throw IpoptException("Not implemented yet!", __FILE__, __LINE__, std::string("OsiIpopt exception"));
}
//---------------------------------------------------------------------------

//***************************************************************************
//***************************************************************************
// OsiConicSolverInterface pure virtual fiunctions
//***************************************************************************
//***************************************************************************
void OsiIpoptSolverInterface::getConicConstraint(int index,
                                                 OsiLorentzConeType & type,
                                                 int & numMembers,
                                                 int *& members) const {
  throw IpoptException("Not implemented yet!", __FILE__, __LINE__, std::string("OsiIpopt exception"));
}

// add conic constraints
// add conic constraint in lorentz cone form
void OsiIpoptSolverInterface::addConicConstraint(OsiLorentzConeType type,
                                                 int numMembers,
                                                 const int * members) {
  if (numCones_>coneMemAllocated_) {
    throw IpoptException("Cone size cannot be greater than size of allocated memory!",
                         __FILE__, __LINE__, std::string("OsiIpopt exception"));
  }
  // check whether we have enough memory allocated
  if (numCones_==coneMemAllocated_) {
    // allocate chunk amount of memory, and copy arrays
    coneMemAllocated_ += CONE_CHUNK;
    int oldNumCones = numCones_;
    int newNumCones = coneMemAllocated_;
    int * newConeSize = new int[newNumCones];
    int * newConeType = new int[newNumCones];
    int ** newConeMembers = new int*[newNumCones];
    // copy cone data to new arrays
    std::copy(coneSize_, coneSize_+oldNumCones, newConeSize);
    std::copy(coneType_, coneType_+oldNumCones, newConeType);
    std::copy(coneMembers_, coneMembers_+oldNumCones, newConeMembers);
    // free old cone data structures
    if (coneSize_) {
      delete[] coneSize_;
      coneSize_ = 0;
    }
    if (coneType_) {
      delete[] coneType_;
      coneType_ = 0;
    }
    if (coneMembers_) {
      delete[] coneMembers_;
      coneMembers_ = 0;
    }
    // assing new cone data
    coneSize_ = newConeSize;
    coneType_ = newConeType;
    coneMembers_ = newConeMembers;
  }
  coneSize_[numCones_] = numMembers;
  if (type==OSI_QUAD) {
    coneType_[numCones_] = 1;
  }
  else if (type==OSI_RQUAD) {
    coneType_[numCones_] = 2;
  }
  else {
    throw IpoptException("!", __FILE__, __LINE__,
                         std::string("Unknown cone type!"));
  }
  coneMembers_[numCones_] = new int[numMembers];
  std::copy(members, members+numMembers, coneMembers_[numCones_]);
  numCones_++;
}

// add conic constraint in |Ax-b| <= dx-h form
void OsiIpoptSolverInterface::addConicConstraint(CoinPackedMatrix const * A,
                                                 CoinPackedVector const * b,
                                                 CoinPackedVector const * d,
                                                 double h) {
  throw IpoptException("Not implemented yet!", __FILE__, __LINE__, std::string("OsiIpopt exception"));
}

void OsiIpoptSolverInterface::removeConicConstraint(int index) {
  throw IpoptException("Not implemented yet!", __FILE__, __LINE__, std::string("OsiIpopt exception"));
}

void OsiIpoptSolverInterface::modifyConicConstraint(int index,
                                                    OsiLorentzConeType type,
                                                    int numMembers,
                                                    const int * members) {
  throw IpoptException("Not implemented yet!", __FILE__, __LINE__, std::string("OsiIpopt exception"));
}

int OsiIpoptSolverInterface::getNumCones() const {
  throw IpoptException("Not implemented yet!", __FILE__, __LINE__, std::string("OsiIpopt exception"));
  return 0;
}

int OsiIpoptSolverInterface::getConeSize(int i) const {
  throw IpoptException("Not implemented yet!", __FILE__, __LINE__, std::string("OsiIpopt exception"));
  return 0;
}

OsiConeType OsiIpoptSolverInterface::getConeType(int i) const {
  throw IpoptException("Not implemented yet!", __FILE__, __LINE__, std::string("OsiIpopt exception"));
  return OSI_LORENTZ;
}

void OsiIpoptSolverInterface::getConeSize(int * size) const {
  throw IpoptException("Not implemented yet!", __FILE__, __LINE__, std::string("OsiIpopt exception"));
}

void OsiIpoptSolverInterface::getConeType(OsiConeType * type) const {
  throw IpoptException("Not implemented yet!", __FILE__, __LINE__, std::string("OsiIpopt exception"));
}

void OsiIpoptSolverInterface::getConeType(OsiLorentzConeType * type) const {
  throw IpoptException("Not implemented yet!", __FILE__, __LINE__, std::string("OsiIpopt exception"));
}

OsiConicSolverInterface * OsiIpoptSolverInterface::clone(bool copyData) const {
  throw IpoptException("Not implemented yet!", __FILE__, __LINE__, std::string("OsiIpopt exception"));
  return 0;
}

//***************************************************************************
//***************************************************************************
// Virtual functions of Ipopt, inherited from TNLP
//***************************************************************************
//***************************************************************************
bool OsiIpoptSolverInterface::get_nlp_info(Index& n, Index& m,
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
bool OsiIpoptSolverInterface::get_bounds_info(Index n, Number* x_l,
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
bool OsiIpoptSolverInterface::get_starting_point(Index n, bool init_x,
                                                 Number* x,
                                                 bool init_z, Number* z_L,
                                                 Number* z_U,
                                                 Index m,
                                                 bool init_lambda,
                                                 Number* lambda) {
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
  return true;
}

/** Method to return the objective value */
bool OsiIpoptSolverInterface::eval_f(Index n, const Number* x,
                                     bool new_x, Number& obj_value) {
  // compute c^Tx
  obj_value = std::inner_product(obj_, obj_+n, x, 0.0);
  return true;
}

/** Method to return the gradient of the objective */
bool OsiIpoptSolverInterface::eval_grad_f(Index n, const Number* x,
                                          bool new_x, Number* grad_f) {
  std::copy(obj_, obj_+n, grad_f);
  return true;
}

/** Method to return the constraint residuals */
bool OsiIpoptSolverInterface::eval_g(Index n, const Number* x,
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
  return true;
}

/** Method to return:
 *   1) The structure of the jacobian (if "values" is NULL)
 *   2) The values of the jacobian (if "values" is not NULL)
 */
bool OsiIpoptSolverInterface::eval_jac_g(Index n, const Number* x, bool new_x,
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
bool OsiIpoptSolverInterface::eval_h(Index n, const Number* x, bool new_x,
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
void OsiIpoptSolverInterface::finalize_solution(
                                     SolverReturn status,
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
  solution_ = new double[n];
  std::copy(x, x+n, solution_);
}

//***************************************************************************
//***************************************************************************
// OsiIpoptSolverInterface methods
//***************************************************************************
//***************************************************************************
// constructors
OsiIpoptSolverInterface::OsiIpoptSolverInterface():
  //  OsiConicSolverInterface(),
  matrix_(0),
  rev_matrix_(0),
  rowlb_(0),
  rowub_(0),
  collb_(0),
  colub_(0),
  obj_(0),
  numCones_(0),
  coneMemAllocated_(0),
  coneSize_(0),
  coneType_(0),
  coneMembers_(0),
  varType_(0),
  app_(0) {
  app_ = IpoptApplicationFactory();
  // Initialize the IpoptApplication and process the options
  //app->Options()->SetIntegerValue("max_iter", 50);
  //app->Options()->SetStringValue("mehrotra_algorithm", "yes");
  Ipopt::ApplicationReturnStatus status;
  status = app_->Initialize();
  if (status != Solve_Succeeded) {
    std::cerr << "OsiIpopt: Error during initialization!" << std::endl;
    throw IpoptException("Error during initialization!", __FILE__,
                         __LINE__, std::string("OsiIpopt exception"));
  }
  // set iteration limit
  //app_->Options()->SetIntegerValue("max_iter", 200);
  //app_->Options()->SetStringValue("mehrotra_algorithm", "yes");
  app_->Options()->SetNumericValue("tol", 1e-5);

}

OsiIpoptSolverInterface::OsiIpoptSolverInterface(
                                  OsiConicSolverInterface const * other) {
  matrix_ = new CoinPackedMatrix(*(other->getMatrixByCol()));
  rev_matrix_ = new CoinPackedMatrix(*(other->getMatrixByRow()));
  int n = matrix_->getNumCols();
  int m = matrix_->getNumRows();
  collb_ = new double[n];
  colub_ = new double[n];
  rowlb_ = new double[m];
  rowub_ = new double[m];
  double const * other_collb = other->getColLower();
  double const * other_colub = other->getColUpper();
  double const * other_rowlb = other->getRowLower();
  double const * other_rowub = other->getRowUpper();
  std::copy(other_collb, other_collb+n, collb_);
  std::copy(other_colub, other_colub+n, colub_);
  std::copy(other_rowlb, other_rowlb+m, rowlb_);
  std::copy(other_rowub, other_rowub+m, rowub_);
  double const * other_obj = other->getObjCoefficients();
  obj_ = new double[n];
  std::copy(other_obj, other_obj+n, obj_);
  numCones_ = 0;
  coneMemAllocated_ = 0;
  coneSize_ = 0;
  coneType_ = 0;
  coneMembers_ = 0;
  varType_ = new VarType[n];
  // todo(aykut) set all variables to continuous for now
  std::fill(varType_, varType_+n, CONTINUOUS);
  int other_num_cones = other->getNumCones();
  for (int i=0; i<other_num_cones; ++i) {
    // get conic constraint i
    OsiLorentzConeType type;
    int num_mem;
    int * members;
    other->getConicConstraint(i, type, num_mem, members);
    // add conic constraint i
    addConicConstraint(type, num_mem, members);
    delete[] members;
  }
  app_ = IpoptApplicationFactory();
  // Initialize the IpoptApplication and process the options
  //app->Options()->SetIntegerValue("max_iter", 50);
  //app->Options()->SetStringValue("mehrotra_algorithm", "yes");
  Ipopt::ApplicationReturnStatus status;
  status = app_->Initialize();
  if (status != Solve_Succeeded) {
    std::cerr << "OsiIpopt: Error during initialization!" << std::endl;
    throw IpoptException("Error during initialization!", __FILE__,
                         __LINE__, std::string("OsiIpopt exception"));
  }
}

// destructor
OsiIpoptSolverInterface::~OsiIpoptSolverInterface() {
  if(matrix_) {
    delete matrix_;
    matrix_ = 0;
  }
  if(rev_matrix_) {
    delete rev_matrix_;
    rev_matrix_ = 0;
  }
  if(rowlb_) {
    delete rowlb_;
    rowlb_ = 0;
  }
  if(rowub_) {
    delete rowub_;
    rowub_ = 0;
  }
  if(collb_) {
    delete collb_;
    collb_ = 0;
  }
  if(colub_) {
    delete colub_;
    colub_ = 0;
  }
  if(obj_) {
    delete obj_;
    obj_ = 0;
  }
  if (coneSize_) {
    delete[] coneSize_;
    coneSize_ = 0;
  }
  if (coneType_) {
    delete[] coneType_;
    coneType_ = 0;
  }
  if (coneMembers_) {
    for (int i=0; i<numCones_; ++i) {
      delete[] coneMembers_[i];
      coneMembers_[i] = 0;
    }
    delete[] coneMembers_;
    coneMembers_ = 0;
  }
  if (varType_) {
    delete[] varType_;
    varType_ = 0;
  }
  // if (app_) {
  //   delete app_;
  //   app_ = 0;
  // }
}

/** Method to return the starting point for the algorithm */
// bool OsiIpoptSolverInterface::mid_point(Index n, bool init_x,
//                                                  Number* x,
//                                                  bool init_z, Number* z_L,
//                                                  Number* z_U,
//                                                  Index m,
//                                                  bool init_lambda,
//                                                  Number* lambda) {
//   assert(init_x == true);
//   assert(init_z == false);
//   assert(init_lambda == false);
//   // set starting point to ub+lb/2
//   for (int i=0; i<n; ++i) {
//     // what if bounds are infinity
//     x[i] = (colub_[i]+collb_[i])/2.0;
//   }
//   // set cone leading variables
//   for (int i=0; i<numCones_; ++i) {
//     // get cone in an array
//     double * par_x = new double[coneSize_[i]];
//     for (int j=0; j<coneSize_[i]; ++j) {
//       par_x[j] = x[coneMembers_[i][j]];
//     }
//     if (coneType_[i]==1) {
//       double ssum = std::inner_product(par_x+1, par_x+coneSize_[i],
//                                        par_x+1, 0.0);
//       x[coneMembers_[i][0]] = sqrt(ssum);
//     }
//     else if (coneType_[i]==2) {
//       double ssum = std::inner_product(par_x+2, par_x+coneSize_[i],
//                                        par_x+2, 0.0);
//       double val = sqrt(ssum/2.0);
//       x[coneMembers_[i][0]] = val;
//       x[coneMembers_[i][1]] = val;
//     }
//   }
//   return true;
// }
