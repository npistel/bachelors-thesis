#include <iostream>
#include <array>
#include <algorithm>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <fstream>
#include <string>
#include <random>
#include <limits>

#include "DisjointSetForest.hpp"
#include "Puzzle.hpp"
#include "State.hpp"

constexpr int m = 3; // rows
constexpr int n = m; // cols
constexpr int N = m * n;

constexpr double EPSILON = 1e-6;

std::size_t pixel_distance(const cv::Vec3b& a, const cv::Vec3b& b)
{
	std::size_t d = 0;

	for (int i = 0; i < 3; i++)
	{
		d += a[i] > b[i] ? a[i] - b[i] : b[i] - a[i];
	}

	return d;
}

std::array<std::size_t, 4> square_distance(const cv::Mat& img, int i, int j)
{
	std::array<std::size_t, 4> d{};

	int i_row = i / n;
	int i_col = i % n;

	int j_row = j / n;
	int j_col = j % n;

	int square_width = img.cols / n;
	int square_height = img.rows / m;

	for (int k = 0; k < square_height; k++)
	{
		d[0] += pixel_distance(img.at<cv::Vec3b>(i_row * square_height + k, (i_col + 1) * square_width - 1), img.at<cv::Vec3b>(j_row * square_height + k, j_col * square_width));
		d[2] += pixel_distance(img.at<cv::Vec3b>(j_row * square_height + k, (j_col + 1) * square_width - 1), img.at<cv::Vec3b>(i_row * square_height + k, i_col * square_width));
	}

	for (int k = 0; k < square_width; k++)
	{
		d[1] += pixel_distance(img.at<cv::Vec3b>((i_row + 1) * square_height - 1, i_col * square_width + k), img.at<cv::Vec3b>(j_row * square_height, j_col * square_width + k));
		d[3] += pixel_distance(img.at<cv::Vec3b>((j_row + 1) * square_height - 1, j_col * square_width + k), img.at<cv::Vec3b>(i_row * square_height, i_col * square_width + k));
	}

	return d;
}

std::vector<std::vector<std::array<std::size_t, 4>>> distance_matrix(const cv::Mat& img)
{
	std::vector<std::vector<std::array<std::size_t, 4>>> M(N, std::vector<std::array<std::size_t, 4>>(N));

	for (int i = 0; i < N - 1; i++)
	{
		for (int j = i + 1; j < N; j++)
		{
			M[i][j] = square_distance(img, i, j);
			M[j][i] = M[i][j];
			std::swap(M[j][i][0], M[j][i][2]);
			std::swap(M[j][i][1], M[j][i][3]);
		}
	}

	return M;
}

cv::Mat reconstruct_image(const cv::Mat& original, const std::vector<std::vector<std::size_t>>& img)
{
	int height = original.rows / m;
	int width = original.cols / n;

	cv::Mat re(img.size() * height, img.front().size() * width, original.type(), cv::Scalar(100, 100, 100));

	for (int i = 0; i < img.size(); i++)
	{
		for (int j = 0; j < img[i].size(); j++)
		{
			if (img[i][j] < N)
			{
				original(
					{ ((int)img[i][j] / n) * height, ((int)img[i][j] / n + 1) * height },
					{ ((int)img[i][j] % n) * width, ((int)img[i][j] % n + 1) * width }
				).copyTo(re(
					{ i * height, (i + 1) * height },
					{ j * width, (j + 1) * width }
				));
			}
		}
	}

	return re;
}

int main(int argc, char* argv[])
{
	cv::Mat img = cv::imread("img/retro_pepe.png");
	if (img.empty())
	{
		std::cout << "could not open image" << std::endl;
		return 0;
	}

	Puzzle p(img, m, n);
	p.shuffle(true);
	
	cv::namedWindow("source_image");
	cv::imshow("source_image", p.image);

	cv::namedWindow("start_state");
	cv::imshow("start_state", p.get_image());

	int hole = p.orig_hole_index;
	auto M = p.distance_matrix();

	std::vector<Edge> edges;
	for (int i = 0; i < N; i++)
	{
		if (i == hole) continue;
		for (int k = 0; k < 4; k++)
		{
			double smallest = std::numeric_limits<double>::max();
			double second_smallest = std::numeric_limits<double>::max();
			for (int j = 0; j < N; j++)
			{
				if (j == i || j == hole) continue;
				if (M[i][j][k] < smallest)
				{
					second_smallest = smallest;
					smallest = M[i][j][k];
				}
				else if (M[i][j][k] < second_smallest)
				{
					second_smallest = M[i][j][k];
				}
			}

			for (int j = 0; j < N; j++)
			{
				if (j == i || j == hole) continue;
				edges.push_back({ i, j, k, M[i][j][k] / (second_smallest + EPSILON) });
			}
		}
	}

	std::sort(edges.begin(), edges.end(), [](const Edge& e1, const Edge& e2) -> bool {
		return e1.weight < e2.weight;
	});

	DisjointSetForest dsf(N);

	for (int i = 0; i < edges.size() && dsf.get_tree_count() > 2; i++)
	{
		dsf.insert_edge(edges[i].v1, edges[i].v2, edges[i].orientation);
	}

	State goal_state = dsf.reconstruct_images(m, n, M).front();

	cv::namedWindow("goal_state");
	cv::imshow("goal_state", p.get_image(goal_state));

	bool solvable = p.state.solveable(goal_state);
	if (!solvable)
	{
		std::cout << "this puzzle is not solvable\n";
	}
	else
	{
		std::cout << "this puzzle is solvable\n";
		std::cout << "starting IDA* search . . .\n";

		bool status = idastar(p.state, goal_state);
		if (!status)
		{
			std::cout << "IDA* search failed . . .\n";
		}
		else
		{
			std::cout << "IDA* search was successful\n";
			auto path = reconstruct_path(goal_state);
			std::cout << "puzzle solvable in at most " << path.size() << " steps\n";
			std::cout << "path: ";
			for (char c : path) std::cout << c << ' ';
			std::cout << "\npress any key to solve . . .\n";

			cv::waitKey();

			for (char c : path)
			{
				if (c == 'w') p.state.slide_up();
				else if (c == 'a') p.state.slide_left();
				else if (c == 's') p.state.slide_down();
				else if (c == 'd') p.state.slide_right();

				cv::imshow("start_state", p.get_image());
				cv::waitKey(100);
			}

			//cv::waitKey();
			//cv::imshow("start_state", img);
		}
	}

	cv::waitKey();
	cv::destroyAllWindows();
}