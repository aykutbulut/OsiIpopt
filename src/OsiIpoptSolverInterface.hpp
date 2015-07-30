// Copyright (C) 2015, Lehigh University All Rights Reserved.
// This code is licensed under the terms of the Eclipse Public License (EPL).

#ifndef OsiIpoptSolverInterface_H
#define OsiIpoptSolverInterface_H

#include <OsiConicSolverInterface.hpp>
#include <IpTNLP.hpp>
#include <IpIpoptApplication.hpp>

typedef enum {
  CONTINUOUS=0,
  BINARY,
  INTEGER
} VarType;

using namespace Ipopt;

class OsiIpoptSolverInterface: virtual public OsiConicSolverInterface,
                               public TNLP {
  CoinPackedMatrix * matrix_;
  double * rowlb_;
  double * rowub_;
  double * collb_;
  double * colub_;
  double * obj_;
  int numCones_;
  int coneMemAllocated_;
  int * coneSize_;
  // 1 for Lorentz cones
  // 2 for rotated Lorentz cones
  int * coneType_;
  int ** coneMembers_;
  // variable types
  VarType * varType_;
  double * solution_;
  IpoptApplication * app_;
public:
  //***************************************************************************
  //***************************************************************************
  // OsiSolverInterface pure virtual fiunctions
  //***************************************************************************
  //***************************************************************************
  ///@name Solve methods
  //@{
  /// Solve initial LP relaxation
  virtual void initialSolve();
  /*! \brief Resolve an LP relaxation after problem modification
    Note the `re-' in `resolve'. initialSolve() should be used to solve the
    problem for the first time.
  */
  virtual void resolve();
  /// Invoke solver's built-in enumeration algorithm
  virtual void branchAndBound();
  //@}
  //---------------------------------------------------------------------------
  /**@name Parameter set/get methods

     The set methods return true if the parameter was set to the given value,
     false otherwise. When a set method returns false, the original value (if
     any) should be unchanged.  There can be various reasons for failure: the
     given parameter is not applicable for the solver (e.g., refactorization
     frequency for the volume algorithm), the parameter is not yet
     implemented for the solver or simply the value of the parameter is out
     of the range the solver accepts. If a parameter setting call returns
     false check the details of your solver.

     The get methods return true if the given parameter is applicable for the
     solver and is implemented. In this case the value of the parameter is
     returned in the second argument. Otherwise they return false.

     \note
     There is a default implementation of the set/get
     methods, namely to store/retrieve the given value using an array in the
     base class. A specific solver implementation can use this feature, for
     example, to store parameters that should be used later on. Implementors
     of a solver interface should overload these functions to provide the
     proper interface to and accurately reflect the capabilities of a
     specific solver.

     The format for hints is slightly different in that a boolean specifies
     the sense of the hint and an enum specifies the strength of the hint.
     Hints should be initialised when a solver is instantiated.
     (See OsiSolverParameters.hpp for defined hint parameters and strength.)
     When specifying the sense of the hint, a value of true means to work with
     the hint, false to work against it.  For example,
     <ul>
     <li> \code setHintParam(OsiDoScale,true,OsiHintTry) \endcode
     is a mild suggestion to the solver to scale the constraint
     system.
     <li> \code setHintParam(OsiDoScale,false,OsiForceDo) \endcode
     tells the solver to disable scaling, or throw an exception if
     it cannot comply.
     </ul>
     As another example, a solver interface could use the value and strength
     of the \c OsiDoReducePrint hint to adjust the amount of information
     printed by the interface and/or solver.  The extent to which a solver
     obeys hints is left to the solver.  The value and strength returned by
     \c getHintParam will match the most recent call to \c setHintParam,
     and will not necessarily reflect the solver's ability to comply with the
     hint.  If the hint strength is \c OsiForceDo, the solver is required to
     throw an exception if it cannot perform the specified action.

     \note
     As with the other set/get methods, there is a default implementation
     which maintains arrays in the base class for hint sense and strength.
     The default implementation does not store the \c otherInformation
     pointer, and always throws an exception for strength \c OsiForceDo.
     Implementors of a solver interface should override these functions to
     provide the proper interface to and accurately reflect the capabilities
     of a specific solver.
  */
  //---------------------------------------------------------------------------
  ///@name Methods returning info on how the solution process terminated
  //@{
  /// Are there numerical difficulties?
  virtual bool isAbandoned() const;
  /// Is optimality proven?
  virtual bool isProvenOptimal() const;
  /// Is primal infeasibility proven?
  virtual bool isProvenPrimalInfeasible() const;
  /// Is dual infeasibility proven?
  virtual bool isProvenDualInfeasible() const;
  /// Is the given primal objective limit reached?
  virtual bool isPrimalObjectiveLimitReached() const;
  /// Is the given dual objective limit reached?
  virtual bool isDualObjectiveLimitReached() const;
  /// Iteration limit reached?
  virtual bool isIterationLimitReached() const;
  //@}

  //---------------------------------------------------------------------------
  /** \name Warm start methods

      Note that the warm start methods return a generic CoinWarmStart object.
      The precise characteristics of this object are solver-dependent. Clients
      who wish to maintain a maximum degree of solver independence should take
      care to avoid unnecessary assumptions about the properties of a warm start
      object.
  */
  //@{
  /*! \brief Get an empty warm start object

    This routine returns an empty warm start object. Its purpose is
    to provide a way for a client to acquire a warm start object of the
    appropriate type for the solver, which can then be resized and modified
    as desired.
  */
  virtual CoinWarmStart *getEmptyWarmStart () const;
  /** \brief Get warm start information.

      Return warm start information for the current state of the solver
      interface. If there is no valid warm start information, an empty warm
      start object wil be returned.
  */
  virtual CoinWarmStart* getWarmStart() const;
  /** \brief Set warm start information.

      Return true or false depending on whether the warm start information was
      accepted or not.
      By definition, a call to setWarmStart with a null parameter should
      cause the solver interface to refresh its warm start information
      from the underlying solver.
  */
  virtual bool setWarmStart(const CoinWarmStart* warmstart);
  //@}

  //---------------------------------------------------------------------------
  /**@name Problem query methods

     Querying a problem that has no data associated with it will result in
     zeros for the number of rows and columns, and NULL pointers from the
     methods that return vectors.

     Const pointers returned from any data-query method are valid as long as
     the data is unchanged and the solver is not called.
  */
  //@{
  /// Get the number of columns
  virtual int getNumCols() const;
  /// Get the number of rows
  virtual int getNumRows() const;
  /// Get the number of nonzero elements
  virtual int getNumElements() const;
  /// Get a pointer to an array[getNumCols()] of column lower bounds
  virtual const double * getColLower() const;
  /// Get a pointer to an array[getNumCols()] of column upper bounds
  virtual const double * getColUpper() const;
  /*! \brief Get a pointer to an array[getNumRows()] of row constraint senses.

    <ul>
    <li>'L': <= constraint
    <li>'E': =  constraint
    <li>'G': >= constraint
    <li>'R': ranged constraint
    <li>'N': free constraint
    </ul>
  */
  virtual const char * getRowSense() const;

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
  virtual const double * getRightHandSide() const;
  /*! \brief Get a pointer to an array[getNumRows()] of row ranges.

    <ul>
    <li> if getRowSense()[i] == 'R' then
    getRowRange()[i] == getRowUpper()[i] - getRowLower()[i]
    <li> if getRowSense()[i] != 'R' then
    getRowRange()[i] is 0.0
    </ul>
  */
  virtual const double * getRowRange() const;
  /// Get a pointer to an array[getNumRows()] of row lower bounds
  virtual const double * getRowLower() const;
  /// Get a pointer to an array[getNumRows()] of row upper bounds
  virtual const double * getRowUpper() const;
  /*! \brief Get a pointer to an array[getNumCols()] of objective
    function coefficients.
  */
  virtual const double * getObjCoefficients() const;
  /*! \brief Get the objective function sense

    -  1 for minimisation (default)
    - -1 for maximisation
  */
  virtual double getObjSense() const;
  /// Return true if the variable is continuous
  virtual bool isContinuous(int colIndex) const;
  /// Return true if the variable is binary
  virtual bool isBinary(int colIndex) const;
  /*! \brief Return true if the variable is integer.

    This method returns true if the variable is binary or general integer.
  */
  virtual bool isInteger(int colIndex) const;
  /// Get a pointer to a row-wise copy of the matrix
  virtual const CoinPackedMatrix * getMatrixByRow() const;
  /// Get a pointer to a column-wise copy of the matrix
  virtual const CoinPackedMatrix * getMatrixByCol() const;
  /// Get the solver's value for infinity
  virtual double getInfinity() const;
  //@}

  /**@name Solution query methods */
  //@{
  /// Get a pointer to an array[getNumCols()] of primal variable values
  virtual const double * getColSolution() const;
  /// Get pointer to array[getNumRows()] of dual variable values
  virtual const double * getRowPrice() const;
  /// Get a pointer to an array[getNumCols()] of reduced costs
  virtual const double * getReducedCost() const;
  /** Get a pointer to array[getNumRows()] of row activity levels.

      The row activity for a row is the left-hand side evaluated at the
      current solution.
  */
  virtual const double * getRowActivity() const;
  /// Get the objective function value.
  virtual double getObjValue() const;
  /** Get the number of iterations it took to solve the problem (whatever
      `iteration' means to the solver).
  */
  virtual int getIterationCount() const;
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
  virtual std::vector<double*> getDualRays(int maxNumRays,
					   bool fullRay = false) const;
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
  virtual std::vector<double*> getPrimalRays(int maxNumRays) const;
  //@}
  //-------------------------------------------------------------------------
  /**@name Methods to modify the objective, bounds, and solution

     For functions which take a set of indices as parameters
     (\c setObjCoeffSet(), \c setColSetBounds(), \c setRowSetBounds(),
     \c setRowSetTypes()), the parameters follow the C++ STL iterator
     convention: \c indexFirst points to the first index in the
     set, and \c indexLast points to a position one past the last index
     in the set.

  */
  //@{
  /** Set an objective function coefficient */
  virtual void setObjCoeff( int elementIndex, double elementValue );
  /** Set the objective function sense.

      Use 1 for minimisation (default), -1 for maximisation.

      \note
      Implementors note that objective function sense is a parameter of
      the OSI, not a property of the problem. Objective sense can be
      set prior to problem load and should not be affected by loading a
      new problem.
  */
  virtual void setObjSense(double s);
  /** Set a single column lower bound.
      Use -getInfinity() for -infinity. */
  virtual void setColLower( int elementIndex, double elementValue );
  /** Set a single column upper bound.
      Use getInfinity() for infinity. */
  virtual void setColUpper( int elementIndex, double elementValue );
  /** Set a single row lower bound.
      Use -getInfinity() for -infinity. */
  virtual void setRowLower( int elementIndex, double elementValue );
  /** Set a single row upper bound.
      Use getInfinity() for infinity. */
  virtual void setRowUpper( int elementIndex, double elementValue );
  /** Set the type of a single row */
  virtual void setRowType(int index, char sense, double rightHandSide,
			  double range);
  /** Set the primal solution variable values

      colsol[getNumCols()] is an array of values for the primal variables.
      These values are copied to memory owned by the solver interface
      object or the solver.  They will be returned as the result of
      getColSolution() until changed by another call to setColSolution() or
      by a call to any solver routine.  Whether the solver makes use of the
      solution in any way is solver-dependent.
  */
  virtual void setColSolution(const double *colsol);
  /** Set dual solution variable values

      rowprice[getNumRows()] is an array of values for the dual variables.
      These values are copied to memory owned by the solver interface
      object or the solver.  They will be returned as the result of
      getRowPrice() until changed by another call to setRowPrice() or by a
      call to any solver routine.  Whether the solver makes use of the
      solution in any way is solver-dependent.
  */
  virtual void setRowPrice(const double * rowprice);
  //@}

  //-------------------------------------------------------------------------
  /**@name Methods to set variable type */
  //@{
  /** Set the index-th variable to be a continuous variable */
  virtual void setContinuous(int index);
  /** Set the index-th variable to be an integer variable */
  virtual void setInteger(int index);
  //@}
  //-------------------------------------------------------------------------

  //-------------------------------------------------------------------------
  /**@name Methods to modify the constraint system.

     Note that new columns are added as continuous variables.
  */
  //@{
  /** Add a column (primal variable) to the problem. */
  virtual void addCol(const CoinPackedVectorBase& vec,
		      const double collb, const double colub,
		      const double obj);
  /** \brief Remove a set of columns (primal variables) from the
      problem.

      The solver interface for a basis-oriented solver will maintain valid
      warm start information if all deleted variables are nonbasic.
  */
  virtual void deleteCols(const int num, const int * colIndices);
  /*! \brief Add a row (constraint) to the problem. */
  virtual void addRow(const CoinPackedVectorBase& vec,
		      const double rowlb, const double rowub);
  /*! \brief Add a row (constraint) to the problem. */
  virtual void addRow(const CoinPackedVectorBase& vec,
		      const char rowsen, const double rowrhs,
		      const double rowrng);
  /** \brief Delete a set of rows (constraints) from the problem.

      The solver interface for a basis-oriented solver will maintain valid
      warm start information if all deleted rows are loose.
  */
  virtual void deleteRows(const int num, const int * rowIndices);
  //@}

  //---------------------------------------------------------------------------

  /**@name Methods for problem input and output */
  //@{
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
  virtual void loadProblem (const CoinPackedMatrix& matrix,
			    const double* collb, const double* colub,
			    const double* obj,
			    const double* rowlb, const double* rowub);
  /*! \brief Load in a problem by assuming ownership of the arguments.
    The constraints on the rows are given by lower and upper bounds.

    For default argument values see the matching loadProblem method.

    \warning
    The arguments passed to this method will be freed using the
    C++ <code>delete</code> and <code>delete[]</code> functions.
  */
  virtual void assignProblem (CoinPackedMatrix*& matrix,
			      double*& collb, double*& colub, double*& obj,
			      double*& rowlb, double*& rowub);
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
  virtual void loadProblem (const CoinPackedMatrix& matrix,
			    const double* collb, const double* colub,
			    const double* obj,
			    const char* rowsen, const double* rowrhs,
			    const double* rowrng);
  /*! \brief Load in a problem by assuming ownership of the arguments.
    The constraints on the rows are given by sense/rhs/range triplets.

    For default argument values see the matching loadProblem method.

    \warning
    The arguments passed to this method will be freed using the
    C++ <code>delete</code> and <code>delete[]</code> functions.
  */
  virtual void assignProblem (CoinPackedMatrix*& matrix,
			      double*& collb, double*& colub, double*& obj,
			      char*& rowsen, double*& rowrhs,
			      double*& rowrng);
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
  virtual void loadProblem (const int numcols, const int numrows,
			    const CoinBigIndex * start, const int* index,
			    const double* value,
			    const double* collb, const double* colub,
			    const double* obj,
			    const double* rowlb, const double* rowub);
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
  virtual void loadProblem (const int numcols, const int numrows,
			    const CoinBigIndex * start, const int* index,
			    const double* value,
			    const double* collb, const double* colub,
			    const double* obj,
			    const char* rowsen, const double* rowrhs,
			    const double* rowrng);
  /*! \brief Write the problem in MPS format to the specified file.

    If objSense is non-zero, a value of -1.0 causes the problem to be
    written with a maximization objective; +1.0 forces a minimization
    objective. If objSense is zero, the choice is left to the implementation.
  */
  virtual void writeMps (const char *filename,
			 const char *extension = "mps",
			 double objSense=0.0) const;
  //---------------------------------------------------------------------------

  ///@name Protected methods of OsiSolverInterface
  //@{
  /** Apply a row cut (append to the constraint matrix). */
  virtual void applyRowCut( const OsiRowCut & rc );
  /** Apply a column cut (adjust the bounds of one or more variables). */
  virtual void applyColCut( const OsiColCut & cc );
  //@}
  //---------------------------------------------------------------------------

  //***************************************************************************
  //***************************************************************************
  // OsiConicSolverInterface pure virtual fiunctions
  //***************************************************************************
  //***************************************************************************
  virtual void getConicConstraint(int index, OsiLorentzConeType & type,
			  int & numMembers,
			  int *& members) const;
  // add conic constraints
  // add conic constraint in lorentz cone form
  virtual void addConicConstraint(OsiLorentzConeType type,
				  int numMembers,
				  const int * members);
  // add conic constraint in |Ax-b| <= dx-h form
  virtual void addConicConstraint(CoinPackedMatrix const * A, CoinPackedVector const * b,
				  CoinPackedVector const * d, double h);
  virtual void removeConicConstraint(int index);
  virtual void modifyConicConstraint(int index, OsiLorentzConeType type,
				     int numMembers,
				     const int * members);
  virtual int getNumCones() const;
  virtual int getConeSize(int i) const;
  virtual OsiConeType getConeType(int i) const;
  virtual void getConeSize(int * size) const;
  virtual void getConeType(OsiConeType * type) const;
  virtual void getConeType(OsiLorentzConeType * type) const;
  virtual OsiConicSolverInterface * clone(bool copyData = true) const;

  //***************************************************************************
  //***************************************************************************
  // Virtual functions of Ipopt, inherited from TNLP
  //***************************************************************************
  //***************************************************************************
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

  //***************************************************************************
  //***************************************************************************
  // OsiIpoptSolverInterface methods
  //***************************************************************************
  //***************************************************************************
  // constructors
  OsiIpoptSolverInterface();
  OsiIpoptSolverInterface(OsiConicSolverInterface const * other);
  // destructor
  virtual ~OsiIpoptSolverInterface();

};

#endif
