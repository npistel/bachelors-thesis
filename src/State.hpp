#pragma once

#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
#include <queue>
#include <map>

class State
{
	public:
		std::size_t rows, cols;
		std::vector<std::size_t> data;
		std::size_t hole_index;

		void permute()
		{
			std::size_t hole_value = this->data[this->hole_index];
			std::next_permutation(this->data.begin(), this->data.end());
			for (this->hole_index = 0; this->hole_index < this->data.size() && this->data[this->hole_index] != hole_value; this->hole_index++) continue;
		}



		std::size_t inversions() const
		{
			std::size_t count = 0;

			for (std::size_t i = 0; i < this->data.size() - 1; i++)
			{
				if (i == this->hole_index) continue;

				for (std::size_t j = i + 1; j < this->data.size(); j++)
				{
					if (j != this->hole_index && this->data[i] > this->data[j])
					{
						count++;
					}
				}
			}

			return count;
		}

		static std::size_t inversion_mergesort(std::vector<std::size_t>& arr, std::vector<std::size_t>& aux, std::size_t off, std::size_t len)
		{
			if (len < 2)
			{
				return 0;
			}

			std::size_t count = inversion_mergesort(arr, aux, off, len / 2) + inversion_mergesort(arr, aux, off + len / 2, (len + 1) / 2);
			for (std::size_t i = 0; i < len; i++)
			{
				aux[i] = arr[off + i];
			}

			for (std::size_t i = 0, j = len / 2, k = off; k < off + len; k++)
			{
				if (i == len / 2 || j < len && aux[i] > aux[j])
				{
					arr[k] = aux[j++];
					count += len / 2 - i;
				}
				else
				{
					arr[k] = aux[i++];
				}
			}

			return count;
		}

		std::size_t inversion_mergesort() const
		{
			auto tmp = this->data;
			tmp.erase(tmp.begin() + this->hole_index);
			auto aux = std::vector<std::size_t>(this->data.size() - 1);
			return inversion_mergesort(tmp, aux, 0, this->data.size() - 1);
		}

		std::size_t inversions(const State& other) const
		{
			if (this->data.size() != other.data.size())
			{
				throw std::exception("permutations must be of equal length");
			}

			std::size_t n = this->data.size();

			std::vector<std::size_t> b_inv(n);
			for (std::size_t i = 0; i < n; i++)
			{
				b_inv[other.data[i]] = i;
			}

			std::vector<std::size_t> a_prime(n);
			for (std::size_t i = 0; i < n; i++)
			{
				a_prime[i] = b_inv[this->data[i]];
			}

			a_prime.erase(a_prime.begin() + this->hole_index); // is there a nicer way to do this?

			return inversion_mergesort(a_prime, b_inv, 0, n - 1);
		}

	public:
		State(const std::vector<std::vector<std::size_t>>& image)
			: rows(image.size())
			, cols(image.front().size())
			, data(rows * cols)
			, hole_index(0)
		{
			std::size_t sum = 0;

			for (std::size_t i = 0; i < this->rows; i++)
			{
				for (std::size_t j = 0; j < this->cols; j++)
				{
					std::size_t k = i * this->cols + j;
					this->data[k] = image[i][j];
					sum += image[i][j];
					if (image[i][j] == this->data.size())
					{
						this->hole_index = k;
					}
				}
			}

			this->data[this->hole_index] = this->data.size() * (this->data.size() + 1) / 2 - sum;
		}

		State(std::size_t rows, std::size_t cols, std::size_t hole_index)
			: rows(rows)
			, cols(cols)
			, data(rows * cols)
			, hole_index(hole_index)
		{
			if (this->hole_index >= this->data.size())
			{
				this->hole_index = this->data.size() - 1;
			}

			for (std::size_t i = 0; i < this->data.size(); i++)
			{
				this->data[i] = i;
			}
		}

		void print() const
		{
			for (std::size_t i = 0; i < this->rows; i++)
			{
				for (std::size_t j = 0; j < this->cols; j++)
				{
					std::size_t k = i * this->cols + j;
					if (k == this->hole_index)
					{
						if (this->data.size() > 10) std::cout << ' ';
						std::cout << 'x';
					}
					else
					{
						if (this->data.size() > 10 && this->data[k] < 10) std::cout << ' ';
						std::cout << this->data[k];
					}

					std::cout << ' ';
				}

				std::cout << '\n';
			}
		}

		bool operator<(const State& other) const
		{
			return std::lexicographical_compare(this->data.begin(), this->data.end(), other.data.begin(), other.data.end());
		}

		bool operator==(const State& other) const
		{
			return std::equal(this->data.begin(), this->data.end(), other.data.begin(), other.data.end());
		}

		bool solveable(const State& other) const
		{
			if (this->rows != other.rows || this->cols != other.cols)
			{
				return false;
			}

			std::size_t n = this->cols;
			// n-1 pieces change relative order with piece that goes into hole
			// even n == parity of k changes
			// odd n == parity of k stays the same

			/*
			std::size_t k1 = this->inversions();
			std::size_t k2 = other.inversions();

			//std::size_t k1 = this->inversion_mergesort();
			//std::size_t k2 = other.inversion_mergesort();

			std::size_t p1 = k1 % 2;
			std::size_t p2 = k2 % 2;

			if (n % 2 == 1) // n odd -> parity cant change -> must already be the same to be solvable
			{
				return p1 == p2;
			}

			// parity changes with every row move
			std::size_t hole_row_1 = this->hole_index / n;
			std::size_t hole_row_2 = other.hole_index / n;

			return (p1 == p2) == (hole_row_1 % 2 == hole_row_2 % 2);
			*/

			std::size_t inv = this->inversions(other);

			if (n % 2 == 1) // n odd -> parity cant change -> must already be even to be solvable
			{
				return inv % 2 == 0;
			}

			// parity changes with every col move
			return (inv % 2 == 0) == ((this->hole_index / n) % 2 == (other.hole_index / n) % 2);
		}

		State shuffle(bool solveable) const
		{
			auto s = *this;
			std::random_device r;
			std::shuffle(s.data.begin(), s.data.end(), r);
			for (s.hole_index = 0; s.hole_index < this->data.size() && s.data[s.hole_index] != this->data[this->hole_index]; s.hole_index++) continue;

			/*
			while (solveable != this->solveable(s) || s.total_manhatten_distance(*this) > 50)
			{
				s.permute();
			}
			return s;
			*/

			if (solveable != this->solveable(s))
			{
				// inversion parity doesnt match
				// change parity of inversions
				// swap two (non hole) elements
				// the transposition will always invert the parity of inversions

				std::size_t i = r() % this->data.size();
				if (i == s.hole_index) i = (i + 1) % this->data.size();

				std::size_t j = r() % this->data.size();

				if (j == s.hole_index || j == i) j = (j + 1) % this->data.size();
				if (j == s.hole_index || j == i) j = (j + 1) % this->data.size();

				std::swap(s.data[i], s.data[j]);
			}

			return s;
		}

		bool slide_down()
		{
			std::size_t row = this->hole_index / this->cols;
			std::size_t col = this->hole_index % this->cols;

			if (row == 0)
			{
				return false;
			}

			std::size_t piece_index = (row - 1) * this->cols + col;
			std::swap(this->data[hole_index], this->data[piece_index]);
			this->hole_index = piece_index;

			return true;
		}

		bool slide_right()
		{
			std::size_t row = this->hole_index / this->cols;
			std::size_t col = this->hole_index % this->cols;

			if (col == 0)
			{
				return false;
			}

			std::size_t piece_index = row * this->cols + col - 1;
			std::swap(this->data[hole_index], this->data[piece_index]);
			this->hole_index = piece_index;

			return true;
		}

		bool slide_up()
		{
			std::size_t row = this->hole_index / this->cols;
			std::size_t col = this->hole_index % this->cols;

			if (row == this->rows - 1)
			{
				return false;
			}

			std::size_t piece_index = (row + 1) * this->cols + col;
			std::swap(this->data[hole_index], this->data[piece_index]);
			this->hole_index = piece_index;

			return true;
		}

		bool slide_left()
		{
			std::size_t row = this->hole_index / this->cols;
			std::size_t col = this->hole_index % this->cols;

			if (col == this->cols - 1)
			{
				return false;
			}

			std::size_t piece_index = row * this->cols + col + 1;
			std::swap(this->data[hole_index], this->data[piece_index]);
			this->hole_index = piece_index;

			return true;
		}

		std::size_t tiles_out_of_place(const State& other) const
		{
			if (this->rows != other.rows || this->cols != other.cols)
			{
				return std::numeric_limits<std::size_t>::max();
			}

			std::size_t count = 0;

			for (std::size_t i = 0; i < this->data.size(); i++)
			{
				count += this->data[i] != other.data[i];
			}

			return count - (this->data[this->hole_index] != other.data[this->hole_index]);
		}

		std::size_t total_manhatten_distance(const State& other) const
		{
			std::size_t sum = 0;

			std::vector<std::size_t> other_inv(this->data.size());
			for (std::size_t i = 0; i < this->data.size(); i++)
			{
				other_inv[other.data[i]] = i;
			}

			for (std::size_t i = 0; i < this->data.size(); i++)
			{
				if (i == this->hole_index) continue;

				std::size_t row1 = i / this->cols;
				std::size_t col1 = i % this->cols;

				std::size_t j = other_inv[this->data[i]];
				std::size_t row2 = j / this->cols;
				std::size_t col2 = j % this->cols;

				sum += std::max(row1, row2) - std::min(row1, row2);
				sum += std::max(col1, col2) - std::min(col1, col2);
			}

			return sum;
		}
};

struct OpenNode
{
	State state;
	State parent;
	std::size_t g;
	std::size_t f;
	char move;

	bool operator<(const OpenNode& other) const
	{
		return this->f > other.f;
	}
};

struct ClosedNode
{
	State parent;
	std::size_t g;
	char move;
};

std::map<State, ClosedNode> closed;

bool bfs(const State& initial_state, const State& end_state)
{
	closed.insert({ initial_state, { initial_state, 0, 'x' } });

	std::queue<OpenNode> open;
	open.push({ initial_state, initial_state, 0, 0, 'x' });

	do
	{
		auto curr_node = open.front();
		open.pop();

		if (curr_node.state == end_state)
		{
			return true;
		}

		auto new_state = curr_node.state;

		if (new_state.slide_down())
		{
			if (closed.count(new_state) == 0)
			{
				closed.insert({ new_state, { curr_node.state, curr_node.g + 1, 's' } });
				open.push({ new_state, curr_node.state, curr_node.g + 1, curr_node.g + 1, 's' });
			}

			new_state.slide_up();
		}

		if (new_state.slide_right())
		{
			if (closed.count(new_state) == 0)
			{
				closed.insert({ new_state, { curr_node.state, curr_node.g + 1, 'd' } });
				open.push({ new_state, curr_node.state, curr_node.g + 1, curr_node.g + 1, 'd' });
			}

			new_state.slide_left();
		}

		if (new_state.slide_up())
		{
			if (closed.count(new_state) == 0)
			{
				closed.insert({ new_state, { curr_node.state, curr_node.g + 1, 'w' } });
				open.push({ new_state, curr_node.state, curr_node.g + 1, curr_node.g + 1, 'w' });
			}

			new_state.slide_down();
		}

		if (new_state.slide_left())
		{
			if (closed.count(new_state) == 0)
			{
				closed.insert({ new_state, { curr_node.state, curr_node.g + 1, 'a' } });
				open.push({ new_state, curr_node.state, curr_node.g + 1, curr_node.g + 1, 'a' });
			}

			new_state.slide_right();
		}
	} while (!open.empty());
	
	return false;
}

void print_path(const State& end_state)
{
	if (closed.empty()) return;
	auto parent = closed.at(end_state);
	if (parent.parent == end_state)
	{
		return;
	}

	print_path(parent.parent);
	std::cout << parent.move << ' ';
}

std::vector<char> reconstruct_path(const State& goal_state)
{
	std::vector<char> path;
	if (closed.empty()) return path;
	
	auto parent_node = closed.at(goal_state);
	if (parent_node.parent == goal_state) return path;

	path = reconstruct_path(parent_node.parent);
	path.push_back(parent_node.move);

	return path;
}

bool a_star(const State& initial_state, const State& end_state)
{
	std::priority_queue<OpenNode> open;
	open.push({ initial_state, initial_state, 0, 0, 'x' });

	do
	{
		auto curr_node = open.top();
		open.pop();

		if (closed.count(curr_node.state) > 0)
		{
			continue;
		}

		closed.insert({ curr_node.state, { curr_node.parent, curr_node.g, curr_node.move } });

		if (curr_node.state == end_state)
		{
			return true;
		}

		auto new_node = curr_node;

		if (new_node.state.slide_down())
		{
			open.push({ new_node.state, curr_node.state, curr_node.g + 1, curr_node.g + 1 + new_node.state.total_manhatten_distance(end_state), 's' });
			new_node.state.slide_up();
		}
		if (new_node.state.slide_right())
		{
			open.push({ new_node.state, curr_node.state, curr_node.g + 1, curr_node.g + 1 + new_node.state.total_manhatten_distance(end_state), 'd' });
			new_node.state.slide_left();
		}
		if (new_node.state.slide_up())
		{
			open.push({ new_node.state, curr_node.state, curr_node.g + 1, curr_node.g + 1 + new_node.state.total_manhatten_distance(end_state), 'w' });
			new_node.state.slide_down();
		}
		if (new_node.state.slide_left())
		{
			open.push({ new_node.state, curr_node.state, curr_node.g + 1, curr_node.g + 1 + new_node.state.total_manhatten_distance(end_state), 'a' });
			new_node.state.slide_right();
		}
	} while (!open.empty());

	return false;
}

std::size_t idastar(const State& start_state, const State& goal_state, std::size_t bound, std::size_t g, char move, const State& parent_state)
{
	std::size_t f = g + start_state.total_manhatten_distance(goal_state);
	if (f > bound) return f;

	if (closed.count(start_state) > 0)
	{
		return std::numeric_limits<std::size_t>::max();
	}

	closed.insert({ start_state, { parent_state, g, move } });

	if (start_state == goal_state) return 0;

	std::size_t min = std::numeric_limits<std::size_t>::max();

	State new_state = start_state;
	if (new_state.slide_down())
	{
		min = std::min(idastar(new_state, goal_state, bound, g + 1, 's', start_state), min);
		if (min == 0) return 0;
		new_state.slide_up();
	}
	if (new_state.slide_right())
	{
		min = std::min(idastar(new_state, goal_state, bound, g + 1, 'd', start_state), min);
		if (min == 0) return 0;
		new_state.slide_left();
	}
	if (new_state.slide_up())
	{
		min = std::min(idastar(new_state, goal_state, bound, g + 1, 'w', start_state), min);
		if (min == 0) return 0;
		new_state.slide_down();
	}
	if (new_state.slide_left())
	{
		min = std::min(idastar(new_state, goal_state, bound, g + 1, 'a', start_state), min);
		if (min == 0) return 0;
		new_state.slide_right();
	}

	closed.erase(start_state);

	return min;
}

bool idastar(const State& start_state, const State& goal_state)
{
	std::size_t bound = start_state.total_manhatten_distance(goal_state);

	do
	{
		closed.clear();

		bound = idastar(start_state, goal_state, bound, 0, 'x', start_state);
		if (bound == 0) return true;
	} while (bound < std::numeric_limits<std::size_t>::max());

	return false;
}