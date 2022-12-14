#ifndef DEMANGLER_H
#define DEMANGLER_H

#include <string>

namespace muq{
namespace Utilities{

  /** Demangles a string returned by type_info::name. */
  std::string demangle(const char* name);

}
}


#endif
