/*
need to read in complex scene from some file format, build the k-d tree, & traverse it when spawning rays
notes for building k-d trees:
    - cuts are made along primary axes (x,y,z)
    - each node contains a bounding box that encompasses all the geometry in its subtree
    - leaf nodes contain a small number of primitives (e.g., triangles)
    - need interior node (subdivision plane axis & value, front & rear children) & leaf node structures (pointer to object(s) found in voxel)
    - decide termination by # of objects in voxel or by depth of tree
    - find partition plane by using spatial median in a round robin fashion
    - partition primitives by testing their bounding boxes against the partition plane
        - primitives can be triangle, AABB, sphere, etc.
        - before you put an object in a k-d tree, but it in a box .. test boxes within boxes
    - needs to be part of world building process, not ray tracing process
    - if interior node, recurse until you get to leaf node, then test ray against all primitives in leaf node
        - will be one of four cases for interior nodes
    - TA-B algorithm makes use of coordinates for entry, exit, & the plane intersection point of axis to traverse the tree efficiently
    - add code to measure the runtime of building the k-d tree & the ray tracing process with & without the k-d tree to show the improvement
*/

// to kick things off:
// all = list of all primitives
// bb = AABB of all primitives
// root = getNode(all, bb)

// Node getNode (List of primitives L, Voxel V) {
//     if (Terminal(L,V)) return new leaf node (L)
//     find partition plane P
//     split V with P producing Vfront & Vrear
//     partition elements of L producing Lfront & Lrear
//     return new interior node (P, getNode(Lfront, Vfront), getNode(Lrear, Vrear))
// }

// bool AABB_intersect (A,B) {
//     for each axis x, y, z:
//     if (Aaxis.min > Baxis.max or Aaxis.max < Baxis.min) return false
//     return true
// }