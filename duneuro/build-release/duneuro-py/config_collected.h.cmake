/* config.h.  Generated from config_collected.h.cmake by CMake.
   It was generated from config_collected.h.cmake which in turn is generated automatically
   from the config.h.cmake files of modules this module depends on. */

/* Define to 1 if you have module duneuro-py available */
#cmakedefine01 HAVE_DUNEURO_PY


/* Define to 1 if you have module dune-common available */
#cmakedefine01 HAVE_DUNE_COMMON


/* Define to 1 if you have module dune-geometry available */
#cmakedefine01 HAVE_DUNE_GEOMETRY


/* Define to 1 if you have module dune-uggrid available */
#cmakedefine01 HAVE_DUNE_UGGRID


/* Define to 1 if you have module dune-grid available */
#cmakedefine01 HAVE_DUNE_GRID


/* Define to 1 if you have module dune-typetree available */
#cmakedefine01 HAVE_DUNE_TYPETREE


/* Define to 1 if you have module dune-istl available */
#cmakedefine01 HAVE_DUNE_ISTL


/* Define to 1 if you have module dune-localfunctions available */
#cmakedefine01 HAVE_DUNE_LOCALFUNCTIONS


/* Define to 1 if you have module dune-alugrid available */
#cmakedefine01 HAVE_DUNE_ALUGRID


/* Define to 1 if you have module dune-functions available */
#cmakedefine01 HAVE_DUNE_FUNCTIONS


/* Define to 1 if you have module dune-mc available */
#cmakedefine01 HAVE_DUNE_MC


/* Define to 1 if you have module dune-subgrid available */
#cmakedefine01 HAVE_DUNE_SUBGRID


/* Define to 1 if you have module dune-udg available */
#cmakedefine01 HAVE_DUNE_UDG


/* Define to 1 if you have module dune-pdelab available */
#cmakedefine01 HAVE_DUNE_PDELAB


/* Define to 1 if you have module duneuro available */
#cmakedefine01 HAVE_DUNEURO


/* begin private */
/* Define to the version of dune-common */
#define DUNE_COMMON_VERSION "${DUNE_COMMON_VERSION}"

/* Define to the major version of dune-common */
#define DUNE_COMMON_VERSION_MAJOR ${DUNE_COMMON_VERSION_MAJOR}

/* Define to the minor version of dune-common */
#define DUNE_COMMON_VERSION_MINOR ${DUNE_COMMON_VERSION_MINOR}

/* Define to the revision of dune-common */
#define DUNE_COMMON_VERSION_REVISION ${DUNE_COMMON_VERSION_REVISION}

/* Standard debug streams with a level below will collapse to doing nothing */
#define DUNE_MINIMAL_DEBUG_LEVEL ${DUNE_MINIMAL_DEBUG_LEVEL}

/* does the compiler support __attribute__((deprecated))? */
#cmakedefine HAS_ATTRIBUTE_DEPRECATED 1

/* does the compiler support __attribute__((deprecated("message"))? */
#cmakedefine HAS_ATTRIBUTE_DEPRECATED_MSG 1

/* does the compiler support __attribute__((unused))? */
#cmakedefine HAS_ATTRIBUTE_UNUSED 1

/* does the standard library provide experimental::make_array() ? */
#cmakedefine DUNE_HAVE_CXX_EXPERIMENTAL_MAKE_ARRAY 1

/* does the standard library provide experimental::is_detected ? */
#cmakedefine DUNE_HAVE_CXX_EXPERIMENTAL_IS_DETECTED 1

/* does the standard library provide identity ? */
#cmakedefine DUNE_HAVE_CXX_STD_IDENTITY 1

/* Define if you have a BLAS library. */
#cmakedefine HAVE_BLAS 1

/* Define if you have LAPACK library. */
#cmakedefine HAVE_LAPACK 1

/* Define if you have the MPI library.  */
#cmakedefine HAVE_MPI ENABLE_MPI

/* Deactivate cxx bindings for MPI */
#if defined(HAVE_MPI) && HAVE_MPI
#define MPICH_SKIP_MPICXX 1
#define OMPI_SKIP_MPICXX 1
#define MPI_NO_CPPBIND 1
#define MPIPP_H
#define _MPICC_H
#endif

/* Define if you have the GNU GMP library. The value should be ENABLE_GMP
   to facilitate activating and deactivating GMP using compile flags. */
#cmakedefine HAVE_GMP ENABLE_GMP

/* Define if you have the GCC Quad-Precision library. The value should be ENABLE_QUADMATH
   to facilitate activating and deactivating QuadMath using compile flags. */
#cmakedefine HAVE_QUADMATH ENABLE_QUADMATH

/* Define if you have the Vc library. The value should be ENABLE_VC
   to facilitate activating and deactivating Vc using compile flags. */
#cmakedefine HAVE_VC ENABLE_VC

/* Define to 1 if you have the Threading Building Blocks (TBB) library */
#cmakedefine HAVE_TBB 1




/* old feature support macros which were tested until 2.7, kept around for one more release */
/* As these are now always supported due to the new compiler requirements, they are directly */
/* defined without an explicit test. */
#define DUNE_HAVE_CXX_CLASS_TEMPLATE_ARGUMENT_DEDUCTION 1
#define DUNE_HAVE_CXX_OPTIONAL 1
#define DUNE_HAVE_CXX_VARIANT 1
#define DUNE_SUPPORTS_CXX_THROW_IN_CONSTEXPR 1
#define DUNE_HAVE_C_ALIGNED_ALLOC 1
#define DUNE_HAVE_CXX_BOOL_CONSTANT 1
#define DUNE_HAVE_CXX_EXPERIMENTAL_BOOL_CONSTANT 0
#define DUNE_HAVE_HEADER_EXPERIMENTAL_TYPE_TRAITS 0
#define DUNE_HAVE_CXX_APPLY 1
#define DUNE_HAVE_CXX_EXPERIMENTAL_APPLY 0
#define HAVE_IS_INDEXABLE_SUPPORT 1

/* Define to ENABLE_UMFPACK if the UMFPack library is available */
#cmakedefine HAVE_UMFPACK ENABLE_SUITESPARSE

/* Define to ENABLE_SUITESPARSE if the SuiteSparse library is available */
#cmakedefine HAVE_SUITESPARSE ENABLE_SUITESPARSE

/* Define to ENABLE_SUITESPARSE if the SuiteSparse's AMD library is available */
#cmakedefine HAVE_SUITESPARSE_AMD ENABLE_SUITESPARSE

/* Define to ENABLE_SUITESPARSE if the SuiteSparse's BTF library is available */
#cmakedefine HAVE_SUITESPARSE_BTF ENABLE_SUITESPARSE

/* Define to ENABLE_SUITESPARSE if the SuiteSparse's CAMD library is available */
#cmakedefine HAVE_SUITESPARSE_CAMD ENABLE_SUITESPARSE

/* Define to ENABLE_SUITESPARSE if the SuiteSparse's CCOLAMD library is available */
#cmakedefine HAVE_SUITESPARSE_CCOLAMD ENABLE_SUITESPARSE

/* Define to ENABLE_SUITESPARSE if the SuiteSparse's CHOLMOD library is available */
#cmakedefine HAVE_SUITESPARSE_CHOLMOD ENABLE_SUITESPARSE

/* Define to ENABLE_SUITESPARSE if the SuiteSparse's COLAMD library is available */
#cmakedefine HAVE_SUITESPARSE_COLAMD ENABLE_SUITESPARSE

/* Define to ENABLE_SUITESPARSE if the SuiteSparse's CXSPARSE library is available */
#cmakedefine HAVE_SUITESPARSE_CXSPARSE ENABLE_SUITESPARSE

/* Define to ENABLE_SUITESPARSE if the SuiteSparse's KLU library is available */
#cmakedefine HAVE_SUITESPARSE_KLU ENABLE_SUITESPARSE

/* Define to ENABLE_SUITESPARSE if the SuiteSparse's LDL library is available */
#cmakedefine HAVE_SUITESPARSE_LDL ENABLE_SUITESPARSE

/* Define to ENABLE_SUITESPARSE if the SuiteSparse's RBIO library is available */
#cmakedefine HAVE_SUITESPARSE_RBIO ENABLE_SUITESPARSE

/* Define to ENABLE_SUITESPARSE if the SuiteSparse's SPQR library is available
   and if it's version is at least 4.3 */
#cmakedefine HAVE_SUITESPARSE_SPQR ENABLE_SUITESPARSE

/* Define to ENABLE_SUITESPARSE if the SuiteSparse's UMFPACK library is available */
#cmakedefine HAVE_SUITESPARSE_UMFPACK ENABLE_SUITESPARSE

/* Define to 1 if METIS is available */
#cmakedefine HAVE_METIS 1

/* Define to 1 if the Scotch replacement for METIS is used. */
#cmakedefine HAVE_SCOTCH_METIS 1

/* Define to 1 if you have the ParMETIS library. */
#cmakedefine HAVE_PARMETIS 1

/* Define to 1 if the PTScotch replacement for ParMETIS is used. */
#cmakedefine HAVE_PTSCOTCH_PARMETIS 1

/* Define to 1 if PT-Scotch is available */
#cmakedefine HAVE_PTSCOTCH 1

/* Used to call lapack functions */
#cmakedefine LAPACK_NEEDS_UNDERLINE





/* Define to the version of dune-geometry */
#define DUNE_GEOMETRY_VERSION "${DUNE_GEOMETRY_VERSION}"

/* Define to the major version of dune-geometry */
#define DUNE_GEOMETRY_VERSION_MAJOR ${DUNE_GEOMETRY_VERSION_MAJOR}

/* Define to the minor version of dune-geometry */
#define DUNE_GEOMETRY_VERSION_MINOR ${DUNE_GEOMETRY_VERSION_MINOR}

/* Define to the revision of dune-geometry */
#define DUNE_GEOMETRY_VERSION_REVISION ${DUNE_GEOMETRY_VERSION_REVISION}



/* Define to the version of dune-common */
#define DUNE_UGGRID_VERSION "${DUNE_UGGRID_VERSION}"

/* Define to the major version of dune-common */
#define DUNE_UGGRID_VERSION_MAJOR ${DUNE_UGGRID_VERSION_MAJOR}

/* Define to the minor version of dune-common */
#define DUNE_UGGRID_VERSION_MINOR ${DUNE_UGGRID_VERSION_MINOR}

/* Define to the revision of dune-common */
#define DUNE_UGGRID_VERSION_REVISION ${DUNE_UGGRID_VERSION_REVISION}

/* begin private section */

/* see parallel/ddd/dddi.h */
#cmakedefine DDD_MAX_PROCBITS_IN_GID ${UG_DDD_MAX_MACROBITS}

/* Define to 1 if you can safely include both <sys/time.h> and <time.h>. */
#cmakedefine TIME_WITH_SYS_TIME 1

/* Define to 1 if UGGrid should use the complete set of green refinement rules for tetrahedra */
#cmakedefine DUNE_UGGRID_TET_RULESET 1

/* end private section */





/* Define to the version of dune-grid */
#define DUNE_GRID_VERSION "${DUNE_GRID_VERSION}"

/* Define to the major version of dune-grid */
#define DUNE_GRID_VERSION_MAJOR ${DUNE_GRID_VERSION_MAJOR}

/* Define to the minor version of dune-grid */
#define DUNE_GRID_VERSION_MINOR ${DUNE_GRID_VERSION_MINOR}

/* Define to the revision of dune-grid */
#define DUNE_GRID_VERSION_REVISION ${DUNE_GRID_VERSION_REVISION}

/* Define to 1 if psurface library is found */
#cmakedefine HAVE_PSURFACE 1

/* Define to 1 if AmiraMesh library is found */
#cmakedefine HAVE_AMIRAMESH 1

/* Define to 1 if you have at least psurface version 2.0 */
#cmakedefine HAVE_PSURFACE_2_0 1

/* Alberta version found by configure, either 0x200 for 2.0 or 0x300 for 3.0 */
#cmakedefine DUNE_ALBERTA_VERSION @DUNE_ALBERTA_VERSION@

/* This is only true if alberta-library was found by configure _and_ if the
   application uses the ALBERTA_CPPFLAGS */
#cmakedefine HAVE_ALBERTA ENABLE_ALBERTA

/* This is only true if UG was found by configure _and_ if the application
   uses the UG_CPPFLAGS */
#cmakedefine HAVE_UG ENABLE_UG

/* Define to 1 if you have mkstemp function */
#cmakedefine01 HAVE_MKSTEMP







/* Define to the version of dune-typetree */
#define DUNE_TYPETREE_VERSION "${DUNE_TYPETREE_VERSION}"

/* Define to the major version of dune-typetree */
#define DUNE_TYPETREE_VERSION_MAJOR ${DUNE_TYPETREE_VERSION_MAJOR}

/* Define to the minor version of dune-typetree */
#define DUNE_TYPETREE_VERSION_MINOR ${DUNE_TYPETREE_VERSION_MINOR}

/* Define to the revision of dune-typetree */
#define DUNE_TYPETREE_VERSION_REVISION ${DUNE_TYPETREE_VERSION_REVISION}






/* Define to ENABLE_SUPERLU if the SuperLU library is available */
#cmakedefine HAVE_SUPERLU ENABLE_SUPERLU

/* Define to the integer type that SuperLU was compiled for
   See e.g. what int_t is defined to in slu_sdefs.h */
#cmakedefine SUPERLU_INT_TYPE @SUPERLU_INT_TYPE@

/* Define to ENABLE_ARPACKPP if the ARPACK++ library is available */
#cmakedefine HAVE_ARPACKPP ENABLE_ARPACKPP

/* Define to the version of dune-istl */
#define DUNE_ISTL_VERSION "${DUNE_ISTL_VERSION}"

/* Define to the major version of dune-istl */
#define DUNE_ISTL_VERSION_MAJOR ${DUNE_ISTL_VERSION_MAJOR}

/* Define to the minor version of dune-istl */
#define DUNE_ISTL_VERSION_MINOR ${DUNE_ISTL_VERSION_MINOR}

/* Define to the revision of dune-istl */
#define DUNE_ISTL_VERSION_REVISION ${DUNE_ISTL_VERSION_REVISION}

/* Enable/Disable the backwards compatibility of the category enum/method in dune-istl solvers, preconditioner, etc. */
#cmakedefine DUNE_ISTL_SUPPORT_OLD_CATEGORY_INTERFACE @DUNE_ISTL_SUPPORT_OLD_CATEGORY_INTERFACE@





/* Define to the version of dune-localfunctions */
#define DUNE_LOCALFUNCTIONS_VERSION "${DUNE_LOCALFUNCTIONS_VERSION}"

/* Define to the major version of dune-localfunctions */
#define DUNE_LOCALFUNCTIONS_VERSION_MAJOR ${DUNE_LOCALFUNCTIONS_VERSION_MAJOR}

/* Define to the minor version of dune-localfunctions */
#define DUNE_LOCALFUNCTIONS_VERSION_MINOR ${DUNE_LOCALFUNCTIONS_VERSION_MINOR}

/* Define to the revision of dune-localfunctions */
#define DUNE_LOCALFUNCTIONS_VERSION_REVISION ${DUNE_LOCALFUNCTIONS_VERSION_REVISION}






#define DUNE_ALUGRID_VERSION "${DUNE_ALUGRID_VERSION}"

/* Define to the major version of dune-alugrid */
#define DUNE_ALUGRID_VERSION_MAJOR ${DUNE_ALUGRID_VERSION_MAJOR}

/* Define to the minor version of dune-alugrid */
#define DUNE_ALUGRID_VERSION_MINOR ${DUNE_ALUGRID_VERSION_MINOR}

/* Define to the revision of dune-alugrid*/
#define DUNE_ALUGRID_VERSION_REVISION ${DUNE_ALUGRID_VERSION_REVISION}

/* Define to build more .cc into library */
#cmakedefine DUNE_ALUGRID_COMPILE_BINDINGS_IN_LIB 1

/* Define if we have dlmalloc */
#cmakedefine HAVE_DLMALLOC 1

/* Define if we have zoltan */
#cmakedefine HAVE_ZOLTAN 1

/* Define if we have ZLIB */
#cmakedefine HAVE_ZLIB 1

/* Include source file for dlmalloc */
#cmakedefine DLMALLOC_SOURCE_INCLUDE ${DLMALLOC_SOURCE_INCLUDE}

/* Define if we have thread local storage */
#cmakedefine HAVE_PTHREAD_TLS 1

/* Define if we have pthreads */
#cmakedefine HAVE_PTHREAD 1

/* Define if testgrids.hh from dune-grid have been found in docs/grids/gridfactory */
#cmakedefine HAVE_DUNE_GRID_TESTGRIDS 1

/* Grid type magic for DGF parser */
@ALUGRID_CONFIG_H_BOTTOM@





/* Define to the version of dune-functions */
#define DUNE_FUNCTIONS_VERSION "@DUNE_FUNCTIONS_VERSION@"

/* Define to the major version of dune-functions */
#define DUNE_FUNCTIONS_VERSION_MAJOR @DUNE_FUNCTIONS_VERSION_MAJOR@

/* Define to the minor version of dune-functions */
#define DUNE_FUNCTIONS_VERSION_MINOR @DUNE_FUNCTIONS_VERSION_MINOR@

/* Define to the revision of dune-functions */
#define DUNE_FUNCTIONS_VERSION_REVISION @DUNE_FUNCTIONS_VERSION_REVISION@





/* Define to the version of dune-pdelab */
#define DUNE_PDELAB_VERSION "${DUNE_PDELAB_VERSION}"

/* Define to the major version of dune-pdelab */
#define DUNE_PDELAB_VERSION_MAJOR ${DUNE_PDELAB_VERSION_MAJOR}

/* Define to the minor version of dune-pdelab */
#define DUNE_PDELAB_VERSION_MINOR ${DUNE_PDELAB_VERSION_MINOR}

/* Define to the revision of dune-pdelab */
#define DUNE_PDELAB_VERSION_REVISION ${DUNE_PDELAB_VERSION_REVISION}

/* This is only true if PETSc was found by configure _and_ if the application
   uses the UG_CPPFLAGS */
#ifndef HAVE_PETSC
#cmakedefine HAVE_PETSC ENABLE_PETSC
#endif

/* This is only true if Eigen3 was found by configure */
#ifndef HAVE_EIGEN
#cmakedefine HAVE_EIGEN ENABLE_EIGEN
#endif

/* Define to 1 if sequential UG has been found */
#cmakedefine PDELAB_SEQUENTIAL_UG 1






/* Define to the version of duneuro */
#define DUNEURO_VERSION "@DUNEURO_VERSION@"

/* Define to the major version of duneuro */
#define DUNEURO_VERSION_MAJOR @DUNEURO_VERSION_MAJOR@

/* Define to the minor version of duneuro */
#define DUNEURO_VERSION_MINOR @DUNEURO_VERSION_MINOR@

/* Define to the revision of duneuro */
#define DUNEURO_VERSION_REVISION @DUNEURO_VERSION_REVISION@

#ifdef HAVE_TBB
#undef HAVE_TBB
#endif
#cmakedefine HAVE_TBB ENABLE_TBB
#cmakedefine HAVE_HDF5WRAP ENABLE_HDF5WRAP
#cmakedefine HAVE_HDF5 ENABLE_HDF5
#cmakedefine HAVE_NIFTI ENABLE_NIFTI



/* begin duneuro-py
   put the definitions for config.h specific to
   your project here. Everything above will be
   overwritten
*/

/* begin private */
/* Name of package */
#define PACKAGE "@DUNE_MOD_NAME@"

/* Define to the address where bug reports for this package should be sent. */
#define PACKAGE_BUGREPORT "@DUNE_MAINTAINER@"

/* Define to the full name of this package. */
#define PACKAGE_NAME "@DUNE_MOD_NAME@"

/* Define to the full name and version of this package. */
#define PACKAGE_STRING "@DUNE_MOD_NAME@ @DUNE_MOD_VERSION@"

/* Define to the one symbol short name of this package. */
#define PACKAGE_TARNAME "@DUNE_MOD_NAME@"

/* Define to the home page for this package. */
#define PACKAGE_URL "@DUNE_MOD_URL@"

/* Define to the version of this package. */
#define PACKAGE_VERSION "@DUNE_MOD_VERSION@"

/* end private */

/* Define to the version of duneuro-py */
#define DUNEURO_PY_VERSION "@DUNEURO_PY_VERSION@"

/* Define to the major version of duneuro-py */
#define DUNEURO_PY_VERSION_MAJOR @DUNEURO_PY_VERSION_MAJOR@

/* Define to the minor version of duneuro-py */
#define DUNEURO_PY_VERSION_MINOR @DUNEURO_PY_VERSION_MINOR@

/* Define to the revision of duneuro-py */
#define DUNEURO_PY_VERSION_REVISION @DUNEURO_PY_VERSION_REVISION@

/* end duneuro-py
   Everything below here will be overwritten
*/ 

/* Grid type magic for DGF parser */
@GRID_CONFIG_H_BOTTOM@

