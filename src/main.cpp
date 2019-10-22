#include <iostream>
#include <array>
#include <algorithm>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <fstream>
#include <string>
#include <random>
#include <limits>
#include <opencv2/imgproc.hpp>

#include "DisjointSetForest.hpp"
#include "Puzzle.hpp"
#include "State.hpp"

constexpr int m = 4; // rows
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

cv::Mat my_get_affine_transform(cv::Point2f u0, cv::Point2f u1, cv::Point2f u2, cv::Point2f v0, cv::Point2f v1, cv::Point2f v2)
{
	/*
	cv::Mat a(3, 3, CV_32F);
	a.at<float>(0, 0) = (u1 - u0).x; a.at<float>(0, 1) = (u2 - u0).x; a.at<float>(0, 2) = u0.x;
	a.at<float>(1, 0) = (u1 - u0).y; a.at<float>(1, 1) = (u2 - u0).y; a.at<float>(1, 2) = u0.y;
	a.at<float>(2, 0) = 0;           a.at<float>(2, 1) = 0;           a.at<float>(2, 2) = 1;

	cv::Mat b(3, 3, CV_32F);
	b.at<float>(0, 0) = (v1 - v0).x; b.at<float>(0, 1) = (v2 - v0).x; b.at<float>(0, 2) = v0.x;
	b.at<float>(1, 0) = (v1 - v0).y; b.at<float>(1, 1) = (v2 - v0).y; b.at<float>(1, 2) = v0.y;
	b.at<float>(2, 0) = 0;           b.at<float>(2, 1) = 0;           b.at<float>(2, 2) = 1;

	auto tmp = b * a.inv();
	return tmp({ 0, 2 }, { 0, 3 });
	*/

	cv::Mat a(3, 3, CV_32F);
	a.at<float>(0, 0) = u0.x; a.at<float>(0, 1) = u1.x; a.at<float>(0, 2) = u2.x;
	a.at<float>(1, 0) = u0.y; a.at<float>(1, 1) = u1.y; a.at<float>(1, 2) = u2.y;
	a.at<float>(2, 0) = 1;    a.at<float>(2, 1) = 1;    a.at<float>(2, 2) = 1;

	cv::Mat b(2, 3, CV_32F);
	b.at<float>(0, 0) = v0.x; b.at<float>(0, 1) = v1.x; b.at<float>(0, 2) = v2.x;
	b.at<float>(1, 0) = v0.y; b.at<float>(1, 1) = v1.y; b.at<float>(1, 2) = v2.y;
	//b.at<float>(2, 0) = 1;    b.at<float>(2, 1) = 1;    b.at<float>(2, 2) = 1;

	auto tmp = b * a.inv();
	return tmp;
}

size_t normal_distance(cv::Mat a, cv::Mat b)
{
	size_t sum = 0;

	for (int i = 0; i < 8; i++)
	{
		for (int ch = 0; ch < 3; ch++)
		{
			int u = a.at<cv::Vec3b>(i, 0)[ch];
			int v = b.at<cv::Vec3b>(i, 0)[ch];
			sum += std::abs(u - v);
		}
	}

	return sum;
}

size_t dtw(cv::Mat a, cv::Mat b)
{
	std::array<std::array<size_t, 8 + 1>, 8 + 1> m{};
	for (auto& n : m) n.fill(std::numeric_limits<size_t>::max());
    m[0][0] = 0;
    
    for (size_t i = 1; i <= 8; i++)
    {
        for (size_t j = 1; j <= 8; j++)
        {
			size_t sum = 0;
			for (size_t ch = 0; ch < 3; ch++)
			{
				int u = a.at<cv::Vec3b>(i - 1, 0)[ch];
				int v = b.at<cv::Vec3b>(j - 1, 0)[ch];
				sum += std::abs(u - v);
			}

            m[i][j] = sum + std::min({
                m[i - 1][j    ],
                m[i    ][j - 1],
                m[i - 1][j - 1]
            });

			int xyz = 5;
        }
    }
    
	return m[8][8];
}

int main(int argc, char* argv[])
{
	cv::Mat a(8, 1, CV_8UC3);
	cv::Mat b(8, 1, CV_8UC3);

	std::srand(3);

	for (int i = 0; i < 8; i++)
	{
		for (int ch = 0; ch < 3; ch++)
		{
			a.at<cv::Vec3b>(i, 0)[ch] = std::rand() % 256;
			b.at<cv::Vec3b>(i, 0)[ch] = std::rand() % 256;
		}
	}

	std::cout << normal_distance(a, b) << std::endl;
	std::cout << dtw(a, b) << std::endl;

	cv::namedWindow("a", cv::WINDOW_FREERATIO);
	cv::imshow("a", a);

	cv::namedWindow("b", cv::WINDOW_FREERATIO);
	cv::imshow("b", b);

	cv::waitKey();
	cv::destroyAllWindows();

	return 0;

	/*-------------------------------------------*/

	cv::Mat src = cv::imread("img/lena2.jpg");

	cv::Mat dst(src.rows + 2, src.cols + 2, src.type(), cv::Scalar(255, 0, 0));
	src.copyTo(dst({1,src.rows+1},{1,src.cols+1}));

	for (int row = 1; row <= src.rows; row++)
	{
		for (int c = 0; c < 3; c++)
		{
			dst.at<cv::Vec3b>(row, src.cols + 1)[c] = cv::saturate_cast<unsigned char>(2 * dst.at<cv::Vec3b>(row, src.cols)[c] - dst.at<cv::Vec3b>(row, src.cols - 1)[c]);
			dst.at<cv::Vec3b>(row, 0)[c] = cv::saturate_cast<unsigned char>(2 * dst.at<cv::Vec3b>(row, 1)[c] - dst.at<cv::Vec3b>(row, 2)[c]);
		}
	}

	for (int col = 1; col <= src.cols; col++)
	{
		for (int c = 0; c < 3; c++)
		{
			dst.at<cv::Vec3b>(src.rows + 1, col)[c] = cv::saturate_cast<unsigned char>(2 * dst.at<cv::Vec3b>(src.rows, col)[c] - dst.at<cv::Vec3b>(src.rows - 1, col)[c]);
			dst.at<cv::Vec3b>(0, col)[c] = cv::saturate_cast<unsigned char>(2 * dst.at<cv::Vec3b>(1, col)[c] - dst.at<cv::Vec3b>(2, col)[c]);
		}
	}

	for (int c = 0; c < 3; c++)
	{
		dst.at<cv::Vec3b>(src.rows + 1, src.cols + 1)[c] = cv::saturate_cast<unsigned char>(
			3 * dst.at<cv::Vec3b>(src.rows, src.cols)[c]
			- dst.at<cv::Vec3b>(src.rows, src.cols - 1)[c]
			- dst.at<cv::Vec3b>(src.rows - 1, src.cols)[c]
		);

		dst.at<cv::Vec3b>(0, src.cols + 1)[c] = cv::saturate_cast<unsigned char>(
			3 * dst.at<cv::Vec3b>(1, src.cols)[c]
			- dst.at<cv::Vec3b>(1, src.cols - 1)[c]
			- dst.at<cv::Vec3b>(2, src.cols)[c]
		);

		dst.at<cv::Vec3b>(src.rows + 1, 0)[c] = cv::saturate_cast<unsigned char>(
			3 * dst.at<cv::Vec3b>(src.rows, 1)[c]
			- dst.at<cv::Vec3b>(src.rows, 2)[c]
			- dst.at<cv::Vec3b>(src.rows - 1, 1)[c]
		);

		dst.at<cv::Vec3b>(0, 0)[c] = cv::saturate_cast<unsigned char>(
			3 * dst.at<cv::Vec3b>(1, 1)[c]
			- dst.at<cv::Vec3b>(1, 2)[c]
			- dst.at<cv::Vec3b>(2, 1)[c]
		);
	}

	cv::namedWindow("src");
	cv::imshow("src", src);

	cv::namedWindow("dst");
	cv::imshow("dst", dst);

	cv::waitKey();
	cv::destroyAllWindows();

	return 0;

	cv::Mat img = cv::imread("img/lena2.jpg");
	if (img.empty())
	{
		std::cout << "could not open image" << std::endl;
		return 0;
	}

	Puzzle p(img, m, n, N-1);
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