#pragma once

#include <algorithm>
#include <vector>

class DisjointSetForest
{
	private:
		struct Coords
		{
			int x, y;
			
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
					return this->y < other.y;
				}

				return this->x < other.x;
			}
		};

		struct TreeNode
		{
			Coords root_coords; // coords of node in root coordinate system
			std::size_t node_index;
		};

		struct DSFNode
		{
			std::size_t parent_index;
			Coords parent_coords; // coords of node in parent coordinate system

			// member only for root-nodes
			std::vector<TreeNode> tree;
		};

		std::vector<DSFNode> nodes;
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
			Coords max = this->nodes[tree_index].tree.back().root_coords;
			max.x = min.x;

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
			}

			Coords d = max - min;
			std::size_t width = d.x + 1;
			std::size_t height = d.y + 1;

			std::vector<std::vector<std::size_t>> img(height, std::vector<std::size_t>(width, this->nodes.size()));

			for (std::size_t i = 0; i < this->nodes[tree_index].tree.size(); i++)
			{
				Coords coords = this->nodes[tree_index].tree[i].root_coords;
				std::size_t row = coords.y - min.y;
				std::size_t col = coords.x - min.x;

				img[row][col] = this->nodes[tree_index].tree[i].node_index;
			}

			return img;
		}

		Coords get_best_frame_location(const std::vector<std::vector<std::size_t>>& image, std::size_t height, std::size_t width) const
		{
			Coords loc = { 0, 0 };
			std::size_t best_count = 0;

			std::size_t count = 0;
			for (std::size_t row = 0; row < height - 1; row++)
			{
				for (std::size_t col = 0; col < width - 1; col++)
				{
					count += image[row][col] < this->nodes.size();
				}
			}

			for (std::size_t row = 0; row < image.size() - height + 1; row++)
			{
				for (std::size_t j = 0; j < width - 1; j++)
				{
					count += image[row + height - 1][j] < this->nodes.size();
				}

				std::size_t row_count = count;
				for (std::size_t col = 0; col < image.front().size() - width + 1; col++)
				{
					for (std::size_t i = 0; i < height; i++)
					{
						row_count += image[row + i][col + width - 1] < this->nodes.size();
					}

					if (row_count > best_count)
					{
						best_count = row_count;
						loc = { static_cast<int>(col), static_cast<int>(row) };
					}

					for (std::size_t i = 0; i < height; i++)
					{
						row_count -= image[row + i][col] < this->nodes.size();
					}
				}

				for (std::size_t j = 0; j < width - 1; j++)
				{
					count -= image[row][j] < this->nodes.size();
				}
			}

			return loc;
		}

	public:
		DisjointSetForest(std::size_t n) : nodes(n), tree_count(n)
		{
			for (std::size_t i = 0; i < n; i++)
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

			// here we are certain that i's tree is no larger than j's
			// we can now append the root of i to the root of j

			static const Coords offsets[] = {
				{ -1,  0 }, // i left  of j
				{  0, -1 }, // i top   of j
				{ +1,  0 }, // i right of j
				{  0, +1 }  // i bot   of j
			};

			Coords offset = offsets[side];
			Coords i_to_j_translation = this->nodes[j].parent_coords + offset - this->nodes[i].parent_coords;
			// note that due to path compression, the parent_coords of each node is the coordinate of that node in its root's coordinate system

			std::vector<TreeNode> transformed_i_nodes;
			std::transform(
				this->nodes[r_i].tree.begin(), this->nodes[r_i].tree.end(),
				std::back_inserter(transformed_i_nodes),
				[i_to_j_translation](const TreeNode& node) -> TreeNode {
					return { node.root_coords + i_to_j_translation, node.node_index };
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

			// check if the intersection is not empty (some pieces overlap when trying to merge)
			if (this->nodes[r_i].tree.size() + this->nodes[r_j].tree.size() - new_nodes.size() > 0)
			{
				return false;
			}

			this->nodes[r_i].parent_coords = i_to_j_translation;
			this->nodes[r_i].parent_index = r_j;
			this->nodes[r_i].tree = std::vector<TreeNode>();

			this->nodes[r_j].tree = new_nodes;

			this->tree_count--;

			return true;
		}

		std::vector<std::vector<std::vector<std::size_t>>> reconstruct_images(std::size_t height, std::size_t width) const
		{
			std::vector<std::vector<std::vector<std::size_t>>> images;

			for (std::size_t i = 0; i < this->nodes.size(); i++)
			{
				if (this->nodes[i].parent_index == i)
				{
					images.emplace_back(height, std::vector<std::size_t>(width, this->nodes.size()));

					auto img = this->reconstruct_image(i);
					height = std::min(height, img.size());
					width = std::min(width, img.front().size());
					Coords loc = this->get_best_frame_location(img, height, width);

					for (std::size_t row = 0; row < height; row++)
					{
						for (std::size_t col = 0; col < width; col++)
						{
							images.back()[row][col] = img[loc.y + row][loc.x + col];
						}
					}

					//images.back() = img;
				}
			}

			return images;
		}
};