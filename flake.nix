{
  description = "Facial Visagism Analysis System";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs";
  };

  outputs = { self, nixpkgs }:
    let
      pkgs = import nixpkgs { system = "x86_64-linux"; };
      python312 = pkgs.python312;
    in
    {
      devShells.x86_64-linux.default = pkgs.mkShell {
        buildInputs = with pkgs; [
          python312
          python312Packages.pip
          python312Packages.numpy
          python312Packages.matplotlib
          (python312Packages.opencv4.override {
            enableGtk2 = true;
          })
          python312Packages.dlib
          python312Packages.pytest
          python312Packages."pytest-cov"
          python312Packages.flake8
          python312Packages.mypy
        ];
      };
    };
}
