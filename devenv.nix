{ pkgs, lib, config, ... }:
let
  pythonPackage = pkgs.python312;
  buildingContainer = config.container.isBuilding;
in
{
  name = "patent-ate";

  env = {
    UV_PYTHON = "${pythonPackage}/bin/python";
    TMPDIR = "${config.env.DEVENV_ROOT}/.cache/tmp";
    UV_CACHE_DIR = "${config.env.DEVENV_ROOT}/.cache/uv";
    XDG_CACHE_HOME = "${config.env.DEVENV_ROOT}/.cache";
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
