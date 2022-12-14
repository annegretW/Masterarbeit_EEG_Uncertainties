#ifndef MULTIINDEXSET_H_
#define MULTIINDEXSET_H_

#include <vector>
#include <memory>
#include <set>
#include <map>

#include "MUQ/Utilities/MultiIndices/MultiIndex.h"

#include "MUQ/Utilities/MultiIndices/MultiIndexLimiter.h"

namespace muq{
  namespace Utilities{

    class H5Object;
    class MultiIndexSet;
    class MultiIndexFactory;


    std::shared_ptr<MultiIndexSet> operator+=( std::shared_ptr<MultiIndexSet> x,
                                               std::shared_ptr<MultiIndexSet> y);

    std::shared_ptr<MultiIndexSet> operator+=( std::shared_ptr<MultiIndexSet> x,
                                               std::shared_ptr<MultiIndex> y);

    /** @class MultiIndexSet
     @ingroup MultiIndices
     @brief A class for holding, sorting, and adapting sets of multiindices.
     @details <p>In the context of polynomial expansions, a multiindex defines a
        single multivariate polynomial.  A finite expansion of multivariate
        polynomials is then defined by a collection of multiindices, one for
        each term in the expansion.  This class is a tool for defining such a
        multiindex set, relating members within the set (i.e. defining
        neighbors), and expanding the set. </p>

     <p>Let \f$\mbox{j}=[j_1,j_2,\dots,j_D]\f$ be a \f$D\f$-dimensional
        multiindex.  The backwards neighbors of \f$\mbox{j}\f$ are the multiindices
        given by multiindices who are only different from \f$\mbox{j}\f$ in one
        component, and in that component, the difference is -1.  For example,
        \f$[j_1-1, j_2,\dots,j_D]\f$ and \f$[j_1, j_2-1,\dots,j_D]\f$ are
        backwards neighbors of \f$\mbox{j}\f$, but \f$[j_1-1,
        j_2-1,\dots,j_D]\f$ and \f$[j_1, j_2-2,\dots,j_D]\f$ are not.  Forward
        neighbors are similarly defined, but with +1. Examples of forward
        neighbors include \f$[j_1+1, j_2,\dots,j_D]\f$ and \f$[j_1,
        j_2+1,\dots,j_D]\f$.   As far as this class is concerned, multiindices
        can be in three different categories: active, inactive, and/or
        admissable.  Active multiindices are those that are currently being used
        to define a polynomial expansion, inactive multiindices are not
        currently used but are being tracked, and admissable multiindices are
        inactive multiindices whose backward neighbors are all active.<p>

     <p>This class keeps track of both active and admissable multiindices, but
        only active indices are included in the linear indexing.  Nonactive
        indices are hidden (i.e. not even considered) in all the public members
        of this class.  For example, the GetAllMultiIndices function will return
        all active multiindices, and the IndexToMulti function will return
        \f$i^\mbox{th}\f$ active multiindex.  Inactive multiindices are used to
        check admissability and are added to the active set during adaptation.
     </p>

     <p>In general, members of the MultiIndexFactory class should be used to
        construct MultiIndexSets directly.<p>
    */
    class MultiIndexSet{

      friend class MultiIndexFactory;

    public:

      MultiIndexSet(const unsigned dimIn,
                    std::shared_ptr<MultiIndexLimiter> limiterIn = std::make_shared<NoLimiter>());

      /// NOTE: does not perform a deep copy of the multiindices themselves, only pointers to the multiindices
      static std::shared_ptr<MultiIndexSet> CloneExisting(std::shared_ptr<MultiIndexSet> const& original);

      /** Default virtual destructor */
      virtual ~MultiIndexSet() = default;

      /** Set the limiter of this MultiIndexSet.  This function will check to make
          sure that all currently active nodes are still feasible with the new limiter.
          If this is not the case, an assert will be thrown.
          @param[in] limiterIn A shared pointer to the new limiter.
      */
      virtual void SetLimiter(std::shared_ptr<MultiIndexLimiter> const& limiterIn);

      /** Returns the limiter used in this MultiIndexSet. */
      virtual std::shared_ptr<MultiIndexLimiter> GetLimiter() const{return limiter;};

      // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
      // * * * FIXED COMPONENTS
      // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *

      /** Get all the active multiindices in this set.
          @return A vector with shared pointers to all the multi indices in the set.
      */
      virtual std::vector<std::shared_ptr<MultiIndex>> GetAllMultiIndices() const {return allMultis;};


      /** Given an index into the set, return the corresponding multiindex as an
          instance of the MultiIndex set. If all the multiindices were stored in a
          vector called multiVec, the functionality of this method would be equivalent
          to multiVec[activeIndex].
          @param[in] activeIndex Linear index of interest.
          @return A shared pointer to a constant instance of the MultiIndex class.
      */
      virtual std::shared_ptr<MultiIndex> const& IndexToMulti(unsigned activeIndex) const{return allMultis.at(active2global.at(activeIndex));};


      /** Given a multiindex, return the linear index where it is located.
          @param[in] input A shared pointer to an instance of the MultiIndex class.
          @return If the multiindex was found in this set, a nonnegative value
          containing the linear index is returned.  However, if the set does not
          contain the multiindex, -1 is returned.
      */
      virtual int MultiToIndex(std::shared_ptr<MultiIndex> const& input) const;

      /** Get the dimension of the multiindex, i.e. how many components does it have? */
      virtual unsigned int GetMultiLength() const{return dim;};

     /** Assume the \f$\mathbf{j}^{\mbox{th}}\f$ multiindex in this set is given by
         \f$\mathbf{j}=[j_1,j_2,\dots,j_D]\f$.  This function returns a vector
         containing the maximum value of any multiindex in each direction, i.e., a
         vector \f$\mathbf{m}=[m_1,m_2,\dots,m_D]\f$ where \f$m_d = \max_{\mathbf{j}}
         j_d\f$.
         @return The vector \f$\mathbf{m}\f$.
     */
      virtual Eigen::VectorXi GetMaxOrders() const{return maxOrders;};

      /**
       * This function provides access to each of the MultiIndices.
       @param[in] activeIndex The index of the active MultiIndex to return.
       @return A pointer to the MultiIndex at index outputIndex.
       */
      virtual std::shared_ptr<MultiIndex> const& at(int activeIndex){return IndexToMulti(activeIndex);}

      /**
       * This function provides access to each of the MultiIndices without any bounds checking on the vector.
       @param[in] outputIndex The index of the active MultiIndex we want to return.
       @return A pointer to the MultiIndex at index outputIndex.
       */
      virtual std::shared_ptr<MultiIndex> operator[](int activeIndex){return allMultis[active2global[activeIndex]]; };

      /**
       * Get the number of active MultiIndices in this set.
       @return An unsigned integer with the number of active MultiIndices in the set.
       */
      virtual unsigned int Size() const{return active2global.size();};

      // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
      // * * * ADAPTIVE COMPONENTS
      // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *

      /** @brief Add another set of multiindices to this one.
          @details Any basis functions in the rhs MultiIndexSet that are not equivalent
                   to basis functions in this instance are added.
          @param[in] rhs Another MultiIndex set to add to this one
          @return A reference to this MultiIndex set, which now contains the union of
                  this set and rhs.
      */
      virtual MultiIndexSet& operator+=(const MultiIndexSet &rhs);

      /** @brief Add a single MultiIndex to the set.
       @details This functions checks to see if the input basis function is
                already in the set and if the input function is unique, it is
                added to the set.
       @param[in] rhs A shared_ptr to the MultiIndex we want to add to the set.
       @return A reference to this MultiIndex set, which may now contain the new
               MultiIndex in rhs.
       */
      virtual MultiIndexSet& operator+=(std::shared_ptr<MultiIndex> const& rhs);

      /** @brief Add all terms in rhs to this instance.
       @details This function adds all unique MultiIndices from the rhs into this MultiIndexSet.  In the event that a multiindex is active in one set, but not the other, the union will set that multiindex to be active.
       @param[in] rhs The MultiIndex set we want to add to this instance.
       @return The number of unique terms from rhs that were added to this set.
       */
      virtual int Union(const MultiIndexSet &rhs);

      /**
         Make the multi-index active. Assumes that the multiIndex input is already
         admissible, as checked by an assertion.  To be admissable (according to
         this function), the multiIndex must already exist as an inactive member
         of this set.  If that is not the case, use the AddActive function instead.
         @param[in] multiIndex A multiindex to make active. Note that an assert will fail if multiIndex is not admissable.
      */
      virtual void Activate(std::shared_ptr<MultiIndex> const& multiIndex);

      /**
       * Add the given multiindex to the set and make it active.  The functionality
         of this function is very similar to Activate; however, this function will
         add the multiIndex if it does not already exist in the set.  This function
         does not check for admissability.  Instead, it will add the multiindex to
         the set and add all neighbors as inactive.  Be careful when using this
         function as it is possible to create a set with active multiindices that
         are not admisable.
       @param[in] newNode A multiindex we want to add as an active member of the set.
       @return An integer specifying the linear index of the now active multiindex.
       */
      virtual int AddActive(std::shared_ptr<MultiIndex> const& newNode);

      /**
         If possible, make the neighbors of this index active, and return any
         that become active.  Do not activate a neighbor that is already part
         of the family.
         @param activeIndex The linear index of the active multiindex to expand.
         @return A vector containing the linear index of any multiindices
                activated because of the expansion.
       */
      virtual std::vector<unsigned> Expand(unsigned int activeIndex);

      /**
         Completely expands an index, whether or not it is currently expandable.
         In order to maintain admissability of the set, it will add backward
         neighbors needed recursively, and return a list of all the indices it adds.
         @param activeIndex The linear index of the active multiindex to expand.
         @return A vector containing the linear index of any multiindices
                 activated because of the expansion.
       */
      virtual std::vector<unsigned> ForciblyExpand(unsigned int const activeIndex);

      /**
       * Add the given multi-index to the active set regardless of whether it's currently admissible.
       * To keep the whole set admissible, recursively add any backward neighbors necessary.
       * Returns a list of indices of any newly added elements, including itself.
       * @param multiIndex The MultiIndex to forcibly add, make active, and make admissable.
       * @return A vector of linear indices indicating all the active MultiIndices
                 added to the set in order to make the given multiIndex admissable.
       */
      virtual std::vector<unsigned> ForciblyActivate(std::shared_ptr<MultiIndex> const& multiIndex);

      /** This function returns the admissable forward neighbors of an active multiindex.
       @param[in] activeIndex The linear index of the active multiIndex under consideration.
       @return A vector of admissible forward neighbors.
       */
      virtual std::vector<std::shared_ptr<MultiIndex>>  GetAdmissibleForwardNeighbors(unsigned int activeIndex);

      /** Here, we define a term on the "frontier" of the multiindex set as one
          that has at least one inactive admissable forward neighbors.  These terms are expandable.
          @return A vector of active "frontier" indices.
      */
      virtual std::vector<unsigned int> GetFrontier() const;

      /** We define the strict frontier to be the collection of multiindices, whose
          forward neighbors are all inactive.
      */
      virtual std::vector<unsigned int> GetStrictFrontier() const;

      /** Returns the indices for the backward neighbors of a currently active multiindex.
      @param[in] activeIndex The linear index of the MultiIndex of interest
      @return A std::vector containing the linear indices of the backward neighbors.
      */
      virtual std::vector<unsigned int> GetBackwardNeighbors(unsigned int activeIndex) const;

      /** Returns indices for backward neighbors of an active or inactive multiindex. */
      virtual std::vector<unsigned int> GetBackwardNeighbors(std::shared_ptr<MultiIndex> const& multiIndex) const;

      //*********************************************************
      //Testing properties of an index/multiIndex
      //*********************************************************

      ///Determines whether the input multiIndex is currently admissible.
      virtual bool IsAdmissible(std::shared_ptr<MultiIndex> const& multiIndex) const;

      ///Return true if one of the forward neighbors of index is admissible but not active.
      virtual bool IsExpandable(unsigned int activeIndex) const;

      ///Return true if the multiIndex is active
      virtual bool IsActive(std::shared_ptr<MultiIndex> const& multiIndex) const;

      /// Returns the number of active forward neighbors
      virtual unsigned int NumActiveForward(unsigned int activeInd) const;

      /// Returns the number of forward neighbors (active or inactive)
      virtual unsigned int NumForward(unsigned int activeInd) const;

      /**
       @brief Saves the multiindex set to a group in an HDF5 file. 
       @details This function will create (or replace) a dataset in an HDF5 file containing the active multiindices in this set.
       Note that all information about the multiindex limiter will be lost when saving.   The dataset defaults to "/multiindices"
       and will contain an \f$N\times D\f$ matrix of integers, where \f$N\f$ is the number of active multiindices in this set and 
       \f$D\f$ is the length of each multiindex.

       @param[in] filename A string to the HDF5 file that this multiindexset should be stored in.   If the file doesn't exist, it will be created.
       @param[in] dsetName The path to the dataset in the HDF5 file where the multiindices will be stored.  If the datset already exists, it will be replaced.   Defaults to "/multiindices".
       */
      void ToHDF5(std::string filename, std::string dsetName="/multiindices") const;

      /** @brief Saves the multiindex set to an HDF5 group object.   
          @details This function will create (or replace) a dataset in an HDF5 file containing the active multiindices in this set.
       Note that all information about the multiindex limiter will be lost when saving.  The dataset will be an \f$N\times D\f$ matrix
       of integers, where \f$N\f$ is the number of active multiindices in this set and  \f$D\f$ is the length of each multiindex.

       ## Typical Usage:
@code{.cpp}
std::shared_ptr<MultiIndexSet> mset = MultiIndexFactory::CreateTotalOrder(2,5);

// Save to HDF5
muq::Utilities::H5Object fout = muq::Utilities::OpenFile("SavedMultis.h5");
mset->ToHDF5(fout, "/SomeGroup/multis");

// Load from HDF5
std::shared_ptr<MultiIndexSet> mset2 = MultiIndexSet::FromHDF5(fout["/SomeGroup/multis"]);
@endcode
          @param[in] group An HDF5 object for the group where the dataset should be created.
          @param[in] dsetName A string containing the name of a new dataset to create inside the group.
      */
      void ToHDF5(muq::Utilities::H5Object &group, std::string dsetName="multiindices") const;

      /**
       @brief Loads a multiindex set from an HDF5 file.  
       @details This function works in tandem with the MultiIndexSet::ToHDF5 function.   It will read the multiindices in the HDF5
       file and return a pointer to a MultiIndexSet object containing those active multiindices.  No limiter information is stored 
       in the HDF5 file, so the limiter in the returned MultiIndexSet will be an instance of the NoLimiter class.

       @param[in] filename A string to an HDF5 file.  If the file doesn't exist, an exception will be thrown.
       @param[in] dsetName The path to the dataset in the HDF5 file containing the multiindices.
       @return std::shared_ptr<MultiIndexSet> 
       */
      static std::shared_ptr<MultiIndexSet> FromHDF5(std::string filename, std::string dsetName="/multiiindices");

      /** @brief Loads the multiindex from an existing HDF5 group.   
          @details This function will read the multiindices in an an HDF5 dataset and construct an instance of the MultiIndexSet class.
          @param[in] dset An HDF5 dataset containing an \f$N\timesD\f$ matrix of non-negative integers defining the multiindices.  Each row is a multiindex.
          
          @see ToHDF5
      */
      static std::shared_ptr<MultiIndexSet> FromHDF5(muq::Utilities::H5Object &dset);

    protected:

      int AddInactive(std::shared_ptr<MultiIndex> const& newNode);

      virtual bool IsAdmissible(unsigned int globalIndex) const;
      virtual bool IsActive(unsigned int globalIndex) const;

      void AddForwardNeighbors(unsigned int globalIndex, bool addInactive);
      void AddBackwardNeighbors(unsigned int globalIndex, bool addInactive);

      void Activate(int globalIndex);
      void ForciblyActivate(int localIndex, std::vector<unsigned int> &newInds);

      // Maps the active index to an entry in allMultis
      std::vector<unsigned> active2global;

      // Maps a global index to an active index.  Non-active values are -1
      std::vector<int> global2active;

      // a vector of sets for the input and output edges.
      std::vector<std::set<int>> outEdges; // edges going out of each multi
      std::vector<std::set<int>> inEdges;  // edges coming in to each multi

      Eigen::VectorXi maxOrders; // the maximum order in each dimension

      // the dimension (i.e., number of components) in each multi index
      unsigned int dim;

      // A vector of both active and admissable multiindices.  Global index.
      std::vector<std::shared_ptr<MultiIndex>> allMultis;

      // store a MultiIndexLimiter that will tell us the feasible set (i.e. simplex, exp limited, etc...)
      std::shared_ptr<MultiIndexLimiter> limiter;

    private:

      int AddMulti(std::shared_ptr<MultiIndex> const& newMulti);

      std::map<std::shared_ptr<MultiIndex>, unsigned int, MultiPtrComp> multi2global; // map from a multiindex to an integer

      MultiIndexSet() = default;

    }; // class MultiIndexSet

  } // namespace Utilities
} // namespace muq




#endif // MULTIINDEXSET_H_
