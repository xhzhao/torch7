package = "torch"
version = "scm-1"

source = {
   url = "git://github.com/torch/torch7.git",
}

description = {
   summary = "Torch7",
   detailed = [[
   ]],
   homepage = "https://github.com/torch/torch7",
   license = "BSD"
}

dependencies = {
   "lua >= 5.1",
   "paths >= 1.0",
   "cwrap >= 1.0"
}

build = {
   type = "command",
   build_command = [[
cmake -E make_directory build && cd build && cmake .. -DCMAKE_BUILD_TYPE=Release -DLUA=$(LUA) -DLUALIB=$(LUALIB) -DLUA_BINDIR="$(LUA_BINDIR)" -DLUA_INCDIR="$(LUA_INCDIR)" -DLUA_LIBDIR="$(LUA_LIBDIR)" -DLUADIR="$(LUADIR)" -DLIBDIR="$(LIBDIR)" -DCMAKE_INSTALL_PREFIX="$(PREFIX)" -DCMAKE_C_COMPILER=icc -DCMAKE_C_FLAGS=-xMIC-AVX512 -DCMAKE_VERBOSE_MAKEFILE=1 -DOpenMP_C_FLAGS=-qopenmp  && $(MAKE) -j$(getconf _NPROCESSORS_ONLN)
]],
	 platforms = {
      windows = {
           build_command = [[
cmake -E make_directory build && cd build && cmake .. -DCMAKE_BUILD_TYPE=Release -DLUA=$(LUA) -DLUALIB=$(LUALIB) -DLUA_BINDIR="$(LUA_BINDIR)" -DLUA_INCDIR="$(LUA_INCDIR)" -DLUA_LIBDIR="$(LUA_LIBDIR)" -DLUADIR="$(LUADIR)" -DLIBDIR="$(LIBDIR)" -DCMAKE_INSTALL_PREFIX="$(PREFIX)" && $(MAKE)
]]
      }
   },
   install_command = "cd build && $(MAKE) install"
}

