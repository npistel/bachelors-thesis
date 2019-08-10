#pragma once

#include <opencv2/core.hpp>
#include <vector>
#include <array>

struct Edge
{
	int v1, v2;
	int orientation;
	double weight;
};

class Puzzle
{
	private:
		struct Piece
		{
			cv::Mat rectangle;
			cv::Matx31d mean[4]; // change back to cv::Vec3d
			cv::Matx33d covar_inv[4];

			static constexpr double dummy_grads[9][3] = {
				{  0,  0,  0 },
				{  1,  1,  1 },
				{ -1, -1, -1 },
				{  0,  0,  1 },
				{  0,  1,  0 },
				{  1,  0,  0 },
				{ -1,  0,  0 },
				{  0, -1,  0 },
				{  0,  0, -1 }
			};

			void compute_mean_and_covar_inv_left()
			{
				cv::Mat_<double> grad(3, this->rectangle.rows + 9);
				for (int i = 0; i < this->rectangle.rows; i++)
				{
					cv::Vec3b p0 = this->rectangle.at<cv::Vec3b>(i, 0);
					cv::Vec3b p1 = this->rectangle.at<cv::Vec3b>(i, 1);

					for (int j = 0; j < 3; j++)
					{
						grad(j, i) = (double)p1[j] - (double)p0[j];
						this->mean[0](j, 0) += grad(j, i);
					}
				}

				this->mean[0] *= 1.0 / this->rectangle.rows;

				for (int i = 0; i < 9; i++)
				{
					for (int j = 0; j < 3; j++)
					{
						grad(j, this->rectangle.rows + i) = dummy_grads[i][j];
					}
				}

				for (int i = 0; i < this->rectangle.rows + 9; i++)
				{
					for (int j = 0; j < 3; j++)
					{
						grad(j, i) -= this->mean[0](j, 0);
					}
				}

				cv::Mat tmp = (grad * grad.t()).inv();
				this->covar_inv[0] = tmp;
			}

			void compute_mean_and_covar_inv_top()
			{
				cv::Mat_<double> grad(3, this->rectangle.cols + 9);
				for (int i = 0; i < this->rectangle.cols; i++)
				{
					cv::Vec3b p0 = this->rectangle.at<cv::Vec3b>(0, i);
					cv::Vec3b p1 = this->rectangle.at<cv::Vec3b>(1, i);

					for (int j = 0; j < 3; j++)
					{
						grad(j, i) = (double)p1[j] - (double)p0[j];
						this->mean[1](j, 0) += grad(j, i);
					}
				}

				this->mean[1] *= 1.0 / this->rectangle.cols;

				for (int i = 0; i < 9; i++)
				{
					for (int j = 0; j < 3; j++)
					{
						grad(j, this->rectangle.cols + i) = dummy_grads[i][j];
					}
				}

				for (int i = 0; i < this->rectangle.cols + 9; i++)
				{
					for (int j = 0; j < 3; j++)
					{
						grad(j, i) -= this->mean[1](j, 0);
					}
				}

				cv::Mat tmp = (grad * grad.t()).inv();
				this->covar_inv[1] = tmp;
			}

			void compute_mean_and_covar_inv_right()
			{
				cv::Mat_<double> grad(3, this->rectangle.rows + 9);
				for (int i = 0; i < this->rectangle.rows; i++)
				{
					cv::Vec3b p0 = this->rectangle.at<cv::Vec3b>(i, this->rectangle.cols - 2);
					cv::Vec3b p1 = this->rectangle.at<cv::Vec3b>(i, this->rectangle.cols - 1);

					for (int j = 0; j < 3; j++)
					{
						grad(j, i) = (double)p1[j] - (double)p0[j];
						this->mean[2](j, 0) += grad(j, i);
					}
				}

				this->mean[2] *= 1.0 / this->rectangle.rows;

				for (int i = 0; i < 9; i++)
				{
					for (int j = 0; j < 3; j++)
					{
						grad(j, this->rectangle.rows + i) = dummy_grads[i][j];
					}
				}

				for (int i = 0; i < this->rectangle.rows + 9; i++)
				{
					for (int j = 0; j < 3; j++)
					{
						grad(j, i) -= this->mean[2](j, 0);
					}
				}

				cv::Mat tmp = (grad * grad.t()).inv();
				this->covar_inv[2] = tmp;
			}

			void compute_mean_and_covar_inv_bot()
			{
				cv::Mat_<double> grad(3, this->rectangle.cols + 9);
				for (int i = 0; i < this->rectangle.cols; i++)
				{
					cv::Vec3b p0 = this->rectangle.at<cv::Vec3b>(this->rectangle.rows - 2, i);
					cv::Vec3b p1 = this->rectangle.at<cv::Vec3b>(this->rectangle.rows - 1, i);

					for (int j = 0; j < 3; j++)
					{
						grad(j, i) = (double)p1[j] - (double)p0[j];
						this->mean[3](j, 0) += grad(j, i);
					}
				}

				this->mean[3] *= 1.0 / this->rectangle.cols;

				for (int i = 0; i < 9; i++)
				{
					for (int j = 0; j < 3; j++)
					{
						grad(j, this->rectangle.cols + i) = dummy_grads[i][j];
					}
				}

				for (int i = 0; i < this->rectangle.cols + 9; i++)
				{
					for (int j = 0; j < 3; j++)
					{
						grad(j, i) -= this->mean[3](j, 0);
					}
				}

				cv::Mat tmp = (grad * grad.t()).inv();
				this->covar_inv[3] = tmp;
			}

			void compute_mean_and_covar_inv()
			{
				compute_mean_and_covar_inv_left();
				compute_mean_and_covar_inv_top();
				compute_mean_and_covar_inv_right();
				compute_mean_and_covar_inv_bot();

				double scale = 1.0;

				this->covar_inv[0] *= 1 / scale;
				this->covar_inv[1] *= 1 / scale;
				this->covar_inv[2] *= 1 / scale;
				this->covar_inv[3] *= 1 / scale;
			}
		};

		cv::Mat image;

		int rows;
		int cols;

		int piece_height;
		int piece_width;

		std::vector<Piece> pieces;

		std::vector<double> pieces_distances(int i, int j) const
		{
			std::vector<double> d(4);

			for (int k = 0; k < this->piece_height; k++)
			{
				cv::Vec3b p0 = this->pieces[i].rectangle.at<cv::Vec3b>(k, this->piece_width - 1);
				cv::Vec3b p1 = this->pieces[j].rectangle.at<cv::Vec3b>(k, 0);

				cv::Matx31d g;
				for (int l = 0; l < 3; l++)
				{
					g(l, 0) = (double)p1(l) - (double)p0(l);
				}

				auto tmp = (g - this->pieces[i].mean[2]).t() * this->pieces[i].covar_inv[2] * (g - this->pieces[i].mean[2]);
				d[0] += tmp(0, 0);

				tmp = (g - this->pieces[j].mean[0]).t() * this->pieces[j].covar_inv[0] * (g - this->pieces[j].mean[0]);
				d[0] += tmp(0, 0);
			}

			for (int k = 0; k < this->piece_height; k++)
			{
				cv::Vec3b p0 = this->pieces[j].rectangle.at<cv::Vec3b>(k, this->piece_width - 1);
				cv::Vec3b p1 = this->pieces[i].rectangle.at<cv::Vec3b>(k, 0);

				cv::Matx31d g;
				for (int l = 0; l < 3; l++)
				{
					g(l, 0) = (double)p1(l) - (double)p0(l);
				}

				auto tmp = (g - this->pieces[i].mean[0]).t() * this->pieces[i].covar_inv[0] * (g - this->pieces[i].mean[0]);
				d[2] += tmp(0, 0);

				tmp = (g - this->pieces[j].mean[2]).t() * this->pieces[j].covar_inv[2] * (g - this->pieces[j].mean[2]);
				d[2] += tmp(0, 0);
			}

			for (int k = 0; k < this->piece_width; k++)
			{
				cv::Vec3b p0 = this->pieces[i].rectangle.at<cv::Vec3b>(this->piece_height - 1, k);
				cv::Vec3b p1 = this->pieces[j].rectangle.at<cv::Vec3b>(0, k);

				cv::Matx31d g;
				for (int l = 0; l < 3; l++)
				{
					g(l, 0) = (double)p1(l) - (double)p0(l);
				}

				auto tmp = (g - this->pieces[i].mean[3]).t() * this->pieces[i].covar_inv[3] * (g - this->pieces[i].mean[3]);
				d[1] += tmp(0, 0);

				tmp = (g - this->pieces[j].mean[1]).t() * this->pieces[j].covar_inv[1] * (g - this->pieces[j].mean[1]);
				d[1] += tmp(0, 0);
			}

			for (int k = 0; k < this->piece_width; k++)
			{
				cv::Vec3b p0 = this->pieces[i].rectangle.at<cv::Vec3b>(0, k);
				cv::Vec3b p1 = this->pieces[j].rectangle.at<cv::Vec3b>(this->piece_height - 1, k);

				cv::Matx31d g;
				for (int l = 0; l < 3; l++)
				{
					g(l, 0) = (double)p1(l) - (double)p0(l);
				}

				auto tmp = (g - this->pieces[i].mean[1]).t() * this->pieces[i].covar_inv[1] * (g - this->pieces[i].mean[1]);
				d[3] += tmp(0, 0);

				tmp = (g - this->pieces[j].mean[3]).t() * this->pieces[j].covar_inv[3] * (g - this->pieces[j].mean[3]);
				d[3] += tmp(0, 0);
			}

			return d;
		}

	public:
		Puzzle(const cv::Mat& image, int rows, int cols)
			: image(image.clone())
			, rows(rows)
			, cols(cols)
			, piece_height(image.rows / rows)
			, piece_width(image.cols / cols)
			, pieces(rows * cols)
		{
			for (int row = 0; row < this->rows; row++)
			{
				for (int col = 0; col < this->cols; col++)
				{
					this->pieces[row * this->cols + col].rectangle = this->image(
						{ row * this->piece_height, (row + 1) * this->piece_height },
						{ col * this->piece_width, (col + 1) * this->piece_width }
					);
				}
			}
		}

		std::vector<std::vector<std::vector<double>>> distance_matrix()
		{
			for (int i = 0; i < this->pieces.size(); i++)
			{
				this->pieces[i].compute_mean_and_covar_inv();
			}

			std::vector<std::vector<std::vector<double>>> m(this->pieces.size(),
				std::vector<std::vector<double>>(this->pieces.size(),
					std::vector<double>(4)
				)
			);

			for (int i = 0; i < this->pieces.size() - 1; i++)
			{
				for (int j = i + 1; j < this->pieces.size(); j++)
				{
					m[i][j] = this->pieces_distances(i, j);
					m[j][i] = m[i][j];
					std::swap(m[j][i][0], m[j][i][2]);
					std::swap(m[j][i][1], m[j][i][3]);
				}
			}

			return m;
		}

		std::vector<std::vector<std::array<double, 4>>> prediction_distance_matrix() const
		{
			std::vector<std::vector<std::array<double, 4>>> m(this->pieces.size(),
				std::vector<std::array<double, 4>>(this->pieces.size())
			);

			static const int L[] = {
				this->piece_height,
				this->piece_width
			};

			for (int i = 0; i < this->pieces.size(); i++)
			{
				for (int j = 0; j < this->pieces.size(); j++)
				{
					if (i == j) continue;

					for (int k = 0; k < 4; k++)
					{
						for (int l = 0; l < L[k % 2]; l++)
						{
							int coords[2][2][2] = {
								{
									{ l, L[(k + 1) % 2] - 1 },
									{ l, L[(k + 1) % 2] - 2 }
								},
								{
									{ l, 0 },
									{ l, 1 }
								}
							};

							cv::Vec3b v_i0 = this->pieces[i].rectangle.at<cv::Vec3b>(coords[k / 2][0][k % 2], coords[k / 2][0][(k + 1) % 2]);
							cv::Vec3b v_i1 = this->pieces[i].rectangle.at<cv::Vec3b>(coords[k / 2][1][k % 2], coords[k / 2][1][(k + 1) % 2]);

							cv::Vec3b v_j0 = this->pieces[j].rectangle.at<cv::Vec3b>(coords[1 - k / 2][0][k % 2], coords[1 - k / 2][0][(k + 1) % 2]);
							cv::Vec3b v_j1 = this->pieces[j].rectangle.at<cv::Vec3b>(coords[1 - k / 2][1][k % 2], coords[1 - k / 2][1][(k + 1) % 2]);

							for (int c = 0; c < 3; c++)
							{
								m[i][j][k] += std::abs(2 * static_cast<double>(v_i0[c]) - static_cast<double>(v_i1[c]) - static_cast<double>(v_j0[c]));
								m[i][j][k] += std::abs(2 * static_cast<double>(v_j0[c]) - static_cast<double>(v_j1[c]) - static_cast<double>(v_i0[c]));
								//m[i][j][k] += std::abs(static_cast<double>(v_i0[c]) - static_cast<double>(v_j0[c]));
							}
						}

						m[j][i][(k + 2) % 4] = m[i][j][k];
					}
				}
			}

			return m;
		}
};