#include <algorithm>
#include <glm/glm.hpp>

using namespace glm;

constexpr const uint W = 1<<12;
constexpr const uint S = W*W;
static_assert(W % 32 == 0);
static_assert(W <= 1<<16);

constexpr const uint BLOCK = 128u;
[[maybe_unused]] constexpr const uint GRID = S / BLOCK;
[[maybe_unused]] constexpr const uint GRID_STRIDED = std::min(128u, GRID);
static_assert(S == GRID * BLOCK);
