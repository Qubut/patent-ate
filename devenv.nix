{ pkgs, lib, config, ... }:
let
  pythonPackage = pkgs.python312;
  buildingContainer = config.container.isBuilding;
  figureTexlive = pkgs.texliveSmall.withPackages (
    ps: with ps; [
      amsfonts
      amsmath
      cm-super
      dvipng
      dvisvgm
      geometry
      pgf
      type1cm
      underscore
    ]
  );
in
{
  name = "patent-ate";

  env = {
    UV_PYTHON = "${pythonPackage}/bin/python";
    TMPDIR = "${config.env.DEVENV_ROOT}/.cache/tmp";
    UV_CACHE_DIR = "${config.env.DEVENV_ROOT}/.cache/uv";
    XDG_CACHE_HOME = "${config.env.DEVENV_ROOT}/.cache";
    LD_LIBRARY_PATH = lib.makeLibraryPath [
      pkgs.zlib
      pkgs.stdenv.cc.cc.lib
    ];
  };

  languages.python = {
    enable = true;
    package = pythonPackage;
    uv.enable = true;
    uv.sync.enable = true;
    venv.enable = true;
  };

  packages = lib.optionals (!buildingContainer) [
    pkgs.git-cliff
    pkgs.uv
    pkgs.ruff
    pkgs.ghostscript
    pkgs.zlib
    figureTexlive
  ];

  containers.prod = {
    name = "patent-ate";
    registry = "docker://ghcr.io/qubut/";
    startupCommand = "patent-ate";
  };

  enterShell = ''
    cache_root="${config.env.DEVENV_ROOT}/.cache"
    mkdir -p "$cache_root/tmp" "$cache_root/uv"
    export TMPDIR="$cache_root/tmp"
    export UV_CACHE_DIR="$cache_root/uv"
    export XDG_CACHE_HOME="$cache_root"
  '';
}
