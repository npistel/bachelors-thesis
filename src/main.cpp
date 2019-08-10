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

constexpr int n = 8;
constexpr int N = n * n;

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
	int square_height = img.rows / n;

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
	std::vector<std::vector<std::array<std::size_t, 4>>> m(N, std::vector<std::array<std::size_t, 4>>(N));

	for (int i = 0; i < N - 1; i++)
	{
		for (int j = i + 1; j < N; j++)
		{
			m[i][j] = square_distance(img, i, j);
			m[j][i] = m[i][j];
			std::swap(m[j][i][0], m[j][i][2]);
			std::swap(m[j][i][1], m[j][i][3]);
		}
	}

	return m;
}

cv::Mat reconstruct_image(const cv::Mat& original, const std::vector<std::vector<std::size_t>>& img)
{
	int height = original.rows / n;
	int width = original.cols / n;

	cv::Mat re(img.size() * height, img.front().size() * width, original.type());

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
	cv::Mat img = cv::imread("img/lena.png");
	if (img.empty())
	{
		std::cout << "ouch" << std::endl;
		return 0;
	}

	Puzzle p(img, n, n);

	auto m = p.distance_matrix();
	//auto m = distance_matrix(img);
	//auto m = p.prediction_distance_matrix();

	std::vector<Edge> edges(N*(N-1)*4);
	int l = 0;
	for (int i = 0; i < N; i++)
	{
		for (int k = 0; k < 4; k++)
		{
			double smallest = std::numeric_limits<double>::max(), second_smallest = std::numeric_limits<double>::max();
			//second_smallest = 1;
			for (int j = 0; j < N; j++)
			{
				//break;
				if (i == j) continue;
				if (m[i][j][k] < smallest)
				{
					second_smallest = smallest;
					smallest = m[i][j][k];
				}
				else if (m[i][j][k] < second_smallest)
				{
					second_smallest = m[i][j][k];
				}
			}

			for (int j = 0; j < N; j++)
			{
				break;
				if (i == j) continue;
				smallest = std::numeric_limits<double>::max();
				for (int ii = 0; ii < N; ii++)
				{
					if (ii != i && m[i][ii][k] < smallest)
					{
						smallest = m[i][ii][k];
					}

					if (ii != j && m[ii][j][k] < smallest)
					{
						smallest = m[ii][j][k];
					}
				}

				edges[l++] = { i, j, k, m[i][j][k] / (smallest + EPSILON) };
			}
			//continue;

			for (int j = 0; j < N; j++)
			{
				if (i == j) continue;
				edges[l++] = { i, j, k, m[i][j][k] / (second_smallest + EPSILON) };
			}
		}
	}

	std::sort(edges.begin(), edges.end(), [](const Edge& e1, const Edge& e2) -> bool {
		return e1.weight < e2.weight;
	});

	DisjointSetForest<N> dsf;

	cv::namedWindow("src", cv::WINDOW_FREERATIO);
	cv::imshow("src", img);
	cv::waitKey();
	//cv::destroyAllWindows();

	/*
	auto images = dsf.reconstruct_images();
	for (int j = 0; j < images.size(); j++)
	{
		auto win_name = std::to_string(j);
		cv::namedWindow(win_name);
		cv::imshow(win_name, reconstruct_image(img, images[j]));
	}

	cv::waitKey();
	cv::destroyAllWindows();
	*/

	do
	{
		for (int i = 0; i < edges.size() && dsf.get_tree_count() > 1; i++)
		{
			if (dsf.insert_edge(edges[i].v1, edges[i].v2, edges[i].orientation))
			{
				continue;
				auto images = dsf.reconstruct_images();
				for (int j = 0; j < images.size(); j++)
				{
					auto win_name = std::to_string(j);
					cv::namedWindow(win_name);
					cv::imshow(win_name, reconstruct_image(img, images[j]));
				}

				//cv::waitKey();
				cv::destroyAllWindows();
			}
		}
	} while (dsf.get_tree_count() > 1);

	auto images = dsf.reconstruct_images();
	for (int j = 0; j < images.size(); j++)
	{
		auto win_name = std::to_string(j);
		cv::namedWindow(win_name, cv::WINDOW_FREERATIO);
		cv::imshow(win_name, reconstruct_image(img, images[j]));
	}

	cv::waitKey();
	cv::destroyAllWindows();

	/*
	std::array<int, N> arr{};
	for (int i = 0; i < N; i++) arr[i] = i;
	std::shuffle(arr.begin(), arr.end(), std::mt19937(std::random_device()()));
	cv::imwrite("bread88.png", arrange(img, arr));
	*/
}