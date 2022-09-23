// three compartement
SetFactory("Built-in");
lc = DefineNumber[ 20, Name "Parameters/lc" ];
z = DefineNumber [127, Name "z"];
c = DefineNumber [127, Name "center"];
// Knoten und Kreise
Point(1) = {c, c, z, lc};
Point(11) = {c-78, c, z, lc};
Point(12) = {c+78, c, z, lc};
Point(21) = {c-86, c, z, lc};
Point(22) = {c+86, c, z, lc};
Point(31) = {c-92, c, z, lc};
Point(32) = {c+92, c, z, lc};
// Point(21) = {c-80, c, z, lc};
// Point(22) = {c+80, c, z, lc};
// Point(31) = {c-86, c, z, lc};
// Point(32) = {c+86, c, z, lc};
// Point(41) = {c-92, c, z, lc};
// Point(42) = {c+92, c, z, lc};
Circle(11) = {11,1,12};
Circle(12) = {12,1,11};
Circle(21) = {21,1,22};
Circle(22) = {22,1,21};
Circle(31) = {31,1,32};
Circle(32) = {32,1,31};
// Circle(41) = {41,1,42};
// Circle(42) = {42,1,41};
// Kreisesegmente verbinden
Curve Loop(1) = {11,12};
Curve Loop(2) = {21,22};
Curve Loop(3) = {31,32};
// Curve Loop(4) = {41,42};
// Flächen definieren
Plane Surface(1) = {1};
Plane Surface(2) = {1, 2};
Plane Surface(3) = {2, 3};
// Plane Surface(4) = {3, 4};
//+
// physical surfaces um die Leitfähigkeiten zuzuweisen
Physical Surface(0) = {1};
Physical Surface(1) = {2};
Physical Surface(2) = {3};
// Physical Surface(4) = {4};
