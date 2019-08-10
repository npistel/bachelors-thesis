#pragma once

#include <algorithm>
#include <array>
#include <vector>

template <std::size_t N>
class DisjointSetForest
{
	private:
		struct Coords
		{
			int x, y; // maybe mirror y-coordinate?
			
			Coords& operator+=(const Coords& other)
			{
				this->x += other.x;
				this->y += other.y;
				return *this;
			}

			Coords operator+(const Coords& other) const
			{
				return Coords(*this) += other;
			}

			Coords operator-() const
			{
				return { -this->x, -this->y };
			}

			Coords& operator-=(const Coords& other)
			{
				return *this += -other;
			}

			Coords operator-(const Coords& other) const
			{
				return Coords(*this) -= other;
			}

			bool operator<(const Coords& other) const
			{
				if (this->y != other.y)
				{
					return this->y > other.y;
				}

				return this->x < other.x;
			}
		};

		struct TreeNode
		{
			Coords root_coords;
			std::size_t node_index;
		};

		struct DSFNode
		{
			std::size_t parent_index;
			Coords parent_coords;

			// member only for root-nodes
			std::vector<TreeNode> tree;
		};

		std::array<DSFNode, N> nodes;
		std::size_t tree_count;

		std::size_t find_root_index(std::size_t node_index)
		{
			if (this->nodes[node_index].parent_index != node_index)
			{
				std::size_t root_index = this->find_root_index(this->nodes[node_index].parent_index);
				this->nodes[node_index].parent_coords += this->nodes[this->nodes[node_index].parent_index].parent_coords;
				this->nodes[node_index].parent_index = root_index;
			}

			return this->nodes[node_index].parent_index;
		}

		std::vector<std::vector<std::size_t>> reconstruct_image(std::size_t tree_index) const
		{
			Coords min = this->nodes[tree_index].tree.front().root_coords;
			Coords max = min;

			for (std::size_t i = 1; i < this->nodes[tree_index].tree.size(); i++)
			{
				Coords coords = this->nodes[tree_index].tree[i].root_coords;

				if (coords.x < min.x)
				{
					min.x = coords.x;
				}
				else if (coords.x > max.x)
				{
					max.x = coords.x;
				}

				if (coords.y < min.y)
				{
					min.y = coords.y;
				}
				else if (coords.y > max.y)
				{
					max.y = coords.y;
				}
			}

			Coords d = max - min;
			std::size_t width = d.x + 1;
			std::size_t height = d.y + 1;

			std::vector<std::vector<std::size_t>> img(height, std::vector<std::size_t>(width, N));

			for (std::size_t i = 0; i < this->nodes[tree_index].tree.size(); i++)
			{
				Coords coords = this->nodes[tree_index].tree[i].root_coords;
				std::size_t row = max.y - coords.y;
				std::size_t col = coords.x - min.x;

				img[row][col] = this->nodes[tree_index].tree[i].node_index;
			}

			return img;
		}

	public:
		DisjointSetForest() : tree_count(N)
		{
			for (std::size_t i = 0; i < N; i++)
			{
				this->nodes[i].parent_index = i;
				this->nodes[i].parent_coords = { 0, 0 };
				this->nodes[i].tree.push_back({ { 0, 0 }, i });
			}
		}

		std::size_t get_tree_count() const
		{
			return this->tree_count;
		}

		bool insert_edge(std::size_t i, std::size_t j, std::size_t side)
		{
			std::size_t r_i = this->find_root_index(i);
			std::size_t r_j = this->find_root_index(j);

			if (r_i == r_j)
			{
				return false;
			}

			if (this->nodes[r_i].tree.size() > this->nodes[r_j].tree.size())
			{
				std::swap(i, j);
				std::swap(r_i, r_j);
				side = (side + 2) % 4;
			}

			static const Coords offsets[] = {
				{ -1,  0 },
				{  0, +1 },
				{ +1,  0 },
				{  0, -1 }
			};

			Coords offset = offsets[side];
			Coords ci_to_cj_translation = this->nodes[j].parent_coords + offset - this->nodes[i].parent_coords;

			std::vector<TreeNode> transformed_i_nodes;
			std::transform(
				this->nodes[r_i].tree.begin(), this->nodes[r_i].tree.end(),
				std::back_inserter(transformed_i_nodes),
				[ci_to_cj_translation](const TreeNode& node) -> TreeNode {
					return { node.root_coords + ci_to_cj_translation, node.node_index };
				}
			);

			std::vector<TreeNode> new_nodes;
			std::set_union(
				transformed_i_nodes.begin(), transformed_i_nodes.end(),
				this->nodes[r_j].tree.begin(), this->nodes[r_j].tree.end(),
				std::back_inserter(new_nodes),
				[](const TreeNode& a, const TreeNode& b) -> bool {
					return a.root_coords < b.root_coords;
				}
			);

			if (this->nodes[r_i].tree.size() + this->nodes[r_j].tree.size() - new_nodes.size() > 0)
			{
				return false;
			}

			this->nodes[r_i].parent_coords = ci_to_cj_translation;
			this->nodes[r_i].parent_index = r_j;
			this->nodes[r_i].tree = std::vector<TreeNode>();

			this->nodes[r_j].tree = new_nodes;

			this->tree_count--;

			return true;
		}

		std::vector<std::vector<std::vector<std::size_t>>> reconstruct_images() const
		{
			std::vector<std::vector<std::vector<std::size_t>>> images;

			for (std::size_t i = 0; i < N; i++)
			{
				if (this->nodes[i].parent_index == i)
				{
					images.push_back(this->reconstruct_image(i));
				}
			}

			return images;
		}
};