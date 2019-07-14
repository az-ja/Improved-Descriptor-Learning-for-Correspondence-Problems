#include "CPM.h"
#include "ImageFeature.h"

#include "opencv2/xfeatures2d.hpp" // for "DAISY" descriptor

#include "tensorflow/cc/ops/const_op.h"
#include "tensorflow/cc/ops/image_ops.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/graph/default_device.h"
#include "tensorflow/core/graph/graph_def_builder.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/util/command_line_flags.h"
#include "tensorflow/contrib/image/kernels/image_ops.h"
#include "tensorflow/contrib/image/kernels/segmentation_ops.h"

// [4/6/2017 Yinlin.Hu]
#include <tuple>
#include <string>
#include <chrono>
#include "math.h"
#define UNKNOWN_FLOW 1e10

#include <algorithm>

template<typename T = int>
class range_2D
{
private:
    bool ended;
    T value;
    T dim0;
    T dim1;

    void test_end()
    {
        if (value >= dim0 * dim1)
            ended = true;
    }

public:
    range_2D()
        : ended(true)
        , dim0(0)
        , dim1(0)
        , value(0)
    {
    }

    range_2D(T end0, T end1, T first0 = 0, T first1 = 0)
        : dim0(end0)
        , dim1(end1)
        , value(first0 + end0 * first1)
    {
        ended = first0 >= dim0 && first1 >= dim1;
    }

    range_2D& operator++()
    {
        if (ended)
            throw runtime_error("ended");
        ++value;
        test_end();
        return *this;
    }

    std::pair<T, T> operator*() const
    {
        if (ended)
            throw runtime_error("ended");
        T first = value % dim0;
        T second = value / dim0;
        return std::make_pair(first, second);
    }

    bool operator!=(range_2D const& other) const
    {
        if (ended && other.ended)
            return false;
        if (ended != other.ended)
            return true;
        else
            return value != other.value;
    }

    bool operator<(range_2D const& other) const
    {
        if (ended && other.ended)
            return false;
        if (ended)
            return false;
        if (other.ended)
            return true;
        else
            return value < other.value;
    }

    range_2D operator+(T offset)
    {
        if (ended)
            throw runtime_error("ended");

        range_2D other(dim0, dim1);
        other.value = value + offset;
        other.test_end();
        return other;
    }

    T operator-(range_2D const& other) const
    {
        if (ended && other.ended)
            return 0;
        else if (ended)
            return other.dim0 * other.dim1 - other.value;
        else if (other.ended)
            return dim0 * dim1 - value;
        else
            return value - other.value;
    }
};

namespace std
{
    template<typename T>
    struct iterator_traits<range_2D<T>> : std::iterator<std::random_access_iterator_tag, std::pair<T, T>> {};
}

template<typename T = int>
range_2D<T> make_start(T end0, T end1)
{
    return range_2D<T>(end0, end1);
}

template<typename T = int>
range_2D<T> make_end(T end0, T end1)
{
    return range_2D<T>(end0, end1, end0, end1);
}

template<typename T = int>
range_2D<T> make_end(T dummy = 0)
{
    return range_2D<T>();
}

using namespace std;
using namespace cv;

using tensorflow::Flag;
using tensorflow::Tensor;
using tensorflow::Status;
using tensorflow::string;
using tensorflow::int32;

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

  string graph = "~/networks_details/output_models/init2_sigmoid_bn_ed500_96_tm_0.0210_10_15_20_25_30_35_40_0000001_5e-06_3/output_graph.pb";
  int const input_width = 16;
  int const input_height = 16;
  int pooling_count = 0;
  string input_layer;
  string output_layer;
  std::unique_ptr<tensorflow::Session> session;
  Status load_graph_status;
  int unwarped_patch_size;
  int half_unwarped_patch_size;

  int toAdd = 0;

 int const layers_count = 7;
 int pooling_order[layers_count] = {0, 0, 0, 0, 0, 0, 1};
 int feat_dim;

 const float mean_R = 92.7982311707;
 const float std_R = 72.3253021888;

 const float mean_G = 88.0847809236;
 const float std_G = 71.2343988627;

 const float mean_B = 77.6385345249;
 const float std_B = 70.0788770213;
Status ReadLabelsFile(const string& file_name, std::vector<string>* result,
                      size_t* found_label_count) {
  std::ifstream file(file_name);
  if (!file) {
    return tensorflow::errors::NotFound("Labels file ", file_name,
                                        " not found.");
  }
  result->clear();
  string line;
  while (std::getline(file, line)) {
    result->push_back(line);
  }
  *found_label_count = result->size();
  const int padding = 16;
  while (result->size() % padding) {
    result->emplace_back();
  }
  return Status::OK();
}

static Status ReadEntireFile(tensorflow::Env* env, const string& filename,
                             Tensor* output) {
  tensorflow::uint64 file_size = 0;
  TF_RETURN_IF_ERROR(env->GetFileSize(filename, &file_size));

  string contents;
  contents.resize(file_size);

  std::unique_ptr<tensorflow::RandomAccessFile> file;
  TF_RETURN_IF_ERROR(env->NewRandomAccessFile(filename, &file));

  tensorflow::StringPiece data;
  TF_RETURN_IF_ERROR(file->Read(0, file_size, &data, &(contents)[0]));
  if (data.size() != file_size) {
    return tensorflow::errors::DataLoss("Truncated read of '", filename,
                                        "' expected ", file_size, " got ",
                                        data.size());
  }
  output->scalar<string>()() = data.ToString();
  return Status::OK();
}

// Given an image file name, read in the data, try to decode it as an image,
// resize it to the requested size, and then scale the values as desired.
Status ReadTensorFromImageFile(const string& file_name, const int input_height,
                               const int input_width, const float input_mean,
                               const float input_std,
                               std::vector<Tensor>* out_tensors) {
  auto root = tensorflow::Scope::NewRootScope();
  using namespace ::tensorflow::ops;  // NOLINT(build/namespaces)

  string input_name = "file_reader";
  string output_name = "normalized";

  // read file_name into a tensor named input
  Tensor input(tensorflow::DT_STRING, tensorflow::TensorShape());
  TF_RETURN_IF_ERROR(
      ReadEntireFile(tensorflow::Env::Default(), file_name, &input));

  // use a placeholder to read input data
  auto file_reader =
      Placeholder(root.WithOpName("input"), tensorflow::DataType::DT_STRING);

  std::vector<std::pair<string, tensorflow::Tensor>> inputs = {
      {"input", input},
  };

  // Now try to figure out what kind of file it is and decode it.
  const int wanted_channels = 3;
  tensorflow::Output image_reader;
  if (tensorflow::str_util::EndsWith(file_name, ".png")) {
    image_reader = DecodePng(root.WithOpName("png_reader"), file_reader,
                             DecodePng::Channels(wanted_channels));
  } else if (tensorflow::str_util::EndsWith(file_name, ".gif")) {
    // gif decoder returns 4-D tensor, remove the first dim
    image_reader =
        Squeeze(root.WithOpName("squeeze_first_dim"),
                DecodeGif(root.WithOpName("gif_reader"), file_reader));
  } else if (tensorflow::str_util::EndsWith(file_name, ".bmp")) {
    image_reader = DecodeBmp(root.WithOpName("bmp_reader"), file_reader);
  } else {
    // Assume if it's neither a PNG nor a GIF then it must be a JPEG.
    image_reader = DecodeJpeg(root.WithOpName("jpeg_reader"), file_reader,
                              DecodeJpeg::Channels(wanted_channels));
  }
  // Now cast the image data to float so we can do normal math on it.
  auto float_caster =
      Cast(root.WithOpName("float_caster"), image_reader, tensorflow::DT_FLOAT);
  // The convention for image ops in TensorFlow is that all images are expected
  // to be in batches, so that they're four-dimensional arrays with indices of
  // [batch, height, width, channel]. Because we only have a single image, we
  // have to add a batch dimension of 1 to the start with ExpandDims().
  auto dims_expander = ExpandDims(root, float_caster, 0);
  // Bilinearly resize the image to fit the required dimensions.
  auto resized = ResizeBilinear(
      root, dims_expander,
      Const(root.WithOpName("size"), {input_height, input_width}));
  // Subtract the mean and divide by the scale.
  Div(root.WithOpName(output_name), Sub(root, resized, {input_mean}),
      {input_std});

  // This runs the GraphDef network definition that we've just constructed, and
  // returns the results in the output tensor.
  tensorflow::GraphDef graph;
  TF_RETURN_IF_ERROR(root.ToGraphDef(&graph));

  std::unique_ptr<tensorflow::Session> session(
      tensorflow::NewSession(tensorflow::SessionOptions()));
  TF_RETURN_IF_ERROR(session->Create(graph));
  TF_RETURN_IF_ERROR(session->Run({inputs}, {output_name}, {}, out_tensors));
  return Status::OK();
}

// Reads a model graph definition from disk, and creates a session object you
// can use to run it.
Status LoadGraph(const string& graph_file_name,
                 std::unique_ptr<tensorflow::Session>* session) {
  tensorflow::GraphDef graph_def;
  Status load_graph_status =
      ReadBinaryProto(tensorflow::Env::Default(), graph_file_name, &graph_def);
  if (!load_graph_status.ok()) {
    return tensorflow::errors::NotFound("Failed to load compute graph at '",
                                        graph_file_name, "'");
  }
  session->reset(tensorflow::NewSession(tensorflow::SessionOptions()));
  Status session_create_status = (*session)->Create(graph_def);
  if (!session_create_status.ok()) {
    return session_create_status;
  }
  return Status::OK();
}

const int halfPatchSize = 8;
const int patchSize = 16;

int unwarped_size(int img_size, int pool_count)
{

 for(int i=0; i<pool_count; i++)
 {
    img_size = ceil(float(img_size/2.0));
 }

 img_size = img_size * pow(2, pool_count);

 return img_size;
}
int w_feats;
int h_feats;


Mat azin_multiPooling(Mat img)
{

  using namespace tensorflow;
  int w = img.cols;
  int h = img.rows;
  int num_channels = 3;
  cout<<"input height and width"<< h << ", "<< w<<endl;


   auto net = [](int dim_init)
  {
  int dim = dim_init;
  int base = 2;
    for(int i=0; i<layers_count; i++)
    {
        if(pooling_order[i] == 1)
        {
            dim -= base;
            dim = dim/2;
        }
        else
        {
        dim -= base;
        }
    }
    return dim * pow(2, pooling_count);
  };

  int extra_needed = (feat_dim -1) * pow(2, pooling_count) ;
  int toBeAdded_w = 0;
  int dim = net(w+ patchSize);
  cout<< "extra needed: "<< extra_needed<< endl;
  cout<< "net(w + patchsize): "<< dim <<endl;

  while(dim - w < extra_needed)
  {
    toBeAdded_w++;
    dim = w + patchSize + toBeAdded_w;
    dim = net(dim);
  }

  int toBeAdded_h = 0;
  dim = net(h + patchSize);

  cout<< "net(h + patchsize): "<< dim <<endl;

  while(dim - h < extra_needed)
  {
  toBeAdded_h ++;
  dim =h + patchSize + toBeAdded_h;
  dim = net(dim);
  }

  cout<<"to be added:"<<toBeAdded_h<< " "<< toBeAdded_w<< endl;


  Mat img_padded(patchSize + h + toBeAdded_h ,patchSize + w + toBeAdded_w,  CV_8UC3, Scalar(0, 0, 0));
  Rect imgBoundaries(halfPatchSize , halfPatchSize , w, h);
  img.copyTo(img_padded(imgBoundaries));
 int  w_ = img_padded.cols;
 int  h_ = img_padded.rows;


  Tensor ten_patch(DT_FLOAT, TensorShape({1 , h_, w_, num_channels}));
  auto tensor_patch = ten_patch.tensor<float, 4>();

// To do normalization per image (npi) :
////mean and std for the img. not the padded:
////float mean_r, mean_g, mean_b, std_r, std_g, std_b = 0;
////    for(int i = 0; i < h; i++)
////    {
////        for(int j = 0; j < w; j++)
////        {
////            for(int c = 0; c < 3; c++)
////            {
////                switch(c)
////                {
////                    case 0:
////                        mean_r += img.at<Vec3b>(i , j)[c] ;
////                        break;
////                    case 1:
////                        mean_g += img.at<Vec3b>(i , j)[c] ;
////                        break;
////                    case 2:
////                        mean_b += img.at<Vec3b>(i , j)[c] ;
////                        break;
////                }
////            }
////        }
////    }
////    int num_pixels = w*h;
////    mean_r /= num_pixels;
////    mean_g /= num_pixels;
////    mean_b /= num_pixels;
////    for(int i = 0; i < h; i++)
////    {
////        for(int j = 0; j < w; j++)
////        {
////            for(int c = 0; c < 3; c++)
////            {
////                switch(c)
////                {
////                    case 0:
////                        std_r += pow(img.at<Vec3b>(i , j)[c] - mean_r, 2);
////                        break;
////                    case 1:
////                        std_g +=  pow(img.at<Vec3b>(i , j)[c] - mean_g, 2);
////                        break;
////                    case 2:
////                        std_b += pow(img.at<Vec3b>(i , j)[c] - mean_b, 2);
////                        break;
////                }
////            }
////        }
////    }
////    std_r /= num_pixels;
////    std_g /= num_pixels;
////    std_b /= num_pixels;
////
////    std_r = sqrt(std_r);
////    std_g = sqrt(std_g);
////    std_b = sqrt(std_b);

    //IMPORTANT: mean_R/std_R (capital r) is for the whole set. Small r is for each image.

  for_each(make_start(h_, w_), make_end(), [&](auto coords)
  {
    int i;
    int j;
    tie(i, j) = coords;
    for(int c=0; c<num_channels; c++)
    {
       tensor_patch(0, i , j , c) = img_padded.at<Vec3b>(i , j)[c];
//To do normalization per whole set (norm):
//        if(c == 0)
//        {
//        tensor_patch(0, i , j , c) = (img_padded.at<Vec3b>(i , j)[c] - mean_R) / std_R;
//        //static_assert(std::is_same<decltype(img_padded.at<Vec3b>(i , j)[c] - mean_R), float>::value, "wrong");
//        }
//        if(c == 1)
//        {
//        tensor_patch(0, i , j , c) = (img_padded.at<Vec3b>(i , j)[c] - mean_G) / std_G;
//        }
//        if(c == 2)
//        {
//        tensor_patch(0, i , j , c) = (img_padded.at<Vec3b>(i , j)[c] - mean_B) / std_B;
//        }
    }

  }, __gnu_parallel::parallel_balanced);
  cout<<"after filling the tensor."<<endl;

// code for non-parallel:
//  for(int i=0 ; i<h_; i++)
//  {
//    for(int j=0; j<w_; j++)
//    {
//        for(int c=0; c<num_channels; c++)
//        {
//            tensor_patch(0, i , j , c) = img_padded.at<Vec3b>(i , j)[c];
//        }
//    }
//  }

  std::vector<Tensor> features;
  Status run_status = session->Run({{input_layer, ten_patch}}, {output_layer}, {}, &features);
  w_feats = features[0].dim_size(1);
  h_feats = features[0].dim_size(0);
  cout<< "new H and width after the net is done:"<< h_feats <<", "<< w_feats<<endl;
  cout<<"vector size:"<<features.size()<<endl;
  int network_depth = features[0].dim_size(2);
  auto tensor_matrix = features[0].tensor<float, 3>();
    Mat out_feats(w*h, network_depth * feat_dim * feat_dim, CV_32FC1);

    for_each(make_start(h, w), make_end(), [&](auto coords)
    {
        int x;
        int y;
        tie(x, y) = coords;
        int it_mat = y + x * w;
        int it_perPixel = -1;
        for(int i = 0; i < feat_dim; i++)
        {
            for(int j = 0; j < feat_dim; j++)
            {
                for (int d = 0; d < network_depth; d++)
                {
                    it_perPixel++;
                    out_feats.at<float>(it_mat ,it_perPixel) = tensor_matrix(x + pow(2, pooling_count) * i, y + pow(2, pooling_count) * j, d);
                }
            }
        }
    }, __gnu_parallel::parallel_balanced);

//code for non-parallel:
//    int it_mat = -1;
//    int it_perPixel;
//    for (int x = 0; x < h; x++)
//    {
//        for (int y = 0; y < w; y++)
//        {
//            it_mat++;
//            it_perPixel = -1;
//            for(int i = 0; i < feat_dim; i++)
//            {
//                for(int j = 0; j < feat_dim; j++)
//                {
//
//                    for (int d = 0; d < network_depth; d++)
//                    {
//                        it_perPixel++;
//                        out_feats.at<float>(it_mat ,it_perPixel) = tensor_matrix(x + pow(2, pooling_count) * i, y + pow(2, pooling_count) * j, d);
//                    }
//
//                }
//            }
//        }
//    }
cout<<"FINISHED"<<endl;

  return out_feats;
}



CPM::CPM()
{
	// default parameters
	_step = 3;
	_isStereo = false;

	_maxIters = 8;
	_stopIterRatio = 0.05;
	_pydRatio = 0.5;

	_maxDisplacement = 400;
	_checkThreshold = 3;
	_borderWidth = 5;

	_im1f = NULL;
	_im2f = NULL;
	_pydSeedsFlow = NULL;
	_pydSeedsFlow2 = NULL;
}

CPM::~CPM()
{
	if (_im1f)
		delete[] _im1f;
	if (_im2f)
		delete[] _im2f;
	if (_pydSeedsFlow)
		delete[] _pydSeedsFlow;
	if (_pydSeedsFlow2)
		delete[] _pydSeedsFlow2;
}

void CPM::SetStereoFlag(int needStereo)
{
	_isStereo = needStereo;
}

void CPM::SetStep(int step)
{
	_step = step;
}

int CPM::Matching(FImage& img1, FImage& img2, FImage& outMatches)
{
    CTimer t;
	int w = img1.width();
	int h = img1.height();
	_pyd1.ConstructPyramid(img1, _pydRatio, 70);
	_pyd2.ConstructPyramid(img2, _pydRatio, 70);

	int nLevels = _pyd1.nlevels();

	if (_im1f)
		delete[] _im1f;
	if (_im2f)
		delete[] _im2f;

	_im1f = new FImage[nLevels];
	_im2f = new FImage[nLevels];
	string method = "Azin";
	if (method == "Azin")
	{
     auto feature_dim = [](int dim_init)
      {
      int dim = dim_init;
      int base = 2;
        for(int i=0; i<layers_count; i++)
        {
            if(pooling_order[i] == 1)
            {
            pooling_count ++;
                dim -= base;
                dim = dim/2;
            }
            else
            {
            dim -= base;
            }
        }
        return dim;
      };
    feat_dim = feature_dim(input_height);
    cout<< "pooling_count: " << pooling_count<<endl;
    cout<< "feat_dim: "<< feat_dim;
    input_layer = "input_node:0";
    output_layer = "NN_model/output_node:0";
    Status load_graph_status = LoadGraph(graph, &session);
    cout<< "load status"<< load_graph_status.ToString()<<endl;

	}
	FImage foo;
	for (int i = 0; i < nLevels; i++){

        if(method=="Azin")
        {
        // in case you want to use separate networks for each pyramid level:
//        if(i==1)
//        {
//        graph = "~/networks_details/output_models/init2_Level1New_sig_bn_ed500_96_tm_0.0210_10_15_20_25_30_35_40_0000001_5e-06_3/output_graph.pb";
//        load_graph_status = LoadGraph(graph, &session);
//        cout<< "load status"<< load_graph_status.ToString()<<endl;
//        }
//        else {
//         if(i>=2)
//        {
//        graph = "~/networks_details/output_models/init2_Level2New_sig_bn_ed500_96_tm_0.0210_10_15_20_25_30_35_40_0000001_5e-06_3/output_graph.pb";
//        load_graph_status = LoadGraph(graph, &session);
//        cout<< "load status"<< load_graph_status.ToString()<<endl;
//       }
//       }

        imAzin(_pyd1[i], _im1f[i]);
        imAzin(_pyd2[i], _im2f[i]);
        }
        else
        {
        imDaisy(_pyd1[i], _im1f[i]);
		imDaisy(_pyd2[i], _im2f[i]);
		//ImageFeature::imSIFT(_pyd1[i], _im1f[i], 2, 1, true, 8);
 		//ImageFeature::imSIFT(_pyd2[i], _im2f[i], 2, 1, true, 8);
        }
	}
	t.toc("get feature: ");

	int step = _step;
	int gridw = w / step;
	int gridh = h / step;
	int xoffset = (w - (gridw - 1)*step) / 2;
	int yoffset = (h - (gridh - 1)*step) / 2;
	int numV = gridw * gridh;
	int numV2 = numV;

	if (_pydSeedsFlow)
		delete[] _pydSeedsFlow;
	if (_pydSeedsFlow2)
		delete[] _pydSeedsFlow2;
	_pydSeedsFlow = new FImage[nLevels];
	_pydSeedsFlow2 = new FImage[nLevels];
	for (int i = 0; i < nLevels; i++){
		_pydSeedsFlow[i].allocate(2, numV);
		_pydSeedsFlow2[i].allocate(2, numV2);
	}

	_seeds.allocate(2, numV);
	_neighbors.allocate(12, numV);
	_neighbors.setValue(-1);
	int nbOffset[8][2] = { { 0, -1 }, { 0, 1 }, { 1, 0 }, { -1, 0 }, { -1, -1 }, { -1, 1 }, { 1, -1 }, { 1, 1 } };
	for (int i = 0; i < numV; i++){
		int gridX = i % gridw;
		int gridY = i / gridw;
		_seeds[2 * i] = gridX * step + xoffset;
		_seeds[2 * i + 1] = gridY * step + yoffset;
		int nbIdx = 0;
		for (int j = 0; j < 8; j++){
			int nbGridX = gridX + nbOffset[j][0];
			int nbGridY = gridY + nbOffset[j][1];
			if (nbGridX < 0 || nbGridX >= gridw || nbGridY < 0 || nbGridY >= gridh)
				continue;
			_neighbors[i*_neighbors.width() + nbIdx] = nbGridY*gridw + nbGridX;
			nbIdx++;
		}
	}
	_seeds2.copy(_seeds);
	_neighbors2.copy(_neighbors);

	FImage seedsFlow(2, numV);

	_kLabels.allocate(w, h);
	for (int i = 0; i < numV; i++){
		int x = _seeds[2 * i];
		int y = _seeds[2 * i + 1];
		int r = step / 2;
		for (int ii = -r; ii <= r; ii++){
			for (int jj = -r; jj <= r; jj++){
				int xx = ImageProcessing::EnforceRange(x + ii, w);
				int yy = ImageProcessing::EnforceRange(y + jj, h);
				_kLabels[yy*w + xx] = i;
			}
		}
	}
	_kLabels2.copy(_kLabels);


	t.tic();
	OnePass(_pyd1, _pyd2, _im1f, _im2f, _seeds, _neighbors, _pydSeedsFlow);
	t.toc("forward matching: ");
	OnePass(_pyd2, _pyd1, _im2f, _im1f, _seeds2, _neighbors2, _pydSeedsFlow2);
	t.toc("backward matching: ");

	// cross check
	int* validFlag = new int[numV];
	CrossCheck(_seeds, _pydSeedsFlow[0], _pydSeedsFlow2[0], _kLabels2, validFlag, _checkThreshold);
	seedsFlow.copyData(_pydSeedsFlow[0]);
	for (int i = 0; i < numV; i++){
		if (!validFlag[i]){
			seedsFlow[2 * i] = UNKNOWN_FLOW;
			seedsFlow[2 * i + 1] = UNKNOWN_FLOW;
		}
	}
	delete[] validFlag;

	// flow 2 match
	FImage tmpMatch(4, numV);
	tmpMatch.setValue(-1);
	int validMatCnt = 0;
	for (int i = 0; i < numV; i++){
		int x = _seeds[2 * i];
		int y = _seeds[2 * i + 1];
		float u = seedsFlow[2 * i];
		float v = seedsFlow[2 * i + 1];
		float x2 = x + u;
		float y2 = y + v;
		if (abs(u) < UNKNOWN_FLOW && abs(v) < UNKNOWN_FLOW){
			tmpMatch[4 * i + 0] = x;
			tmpMatch[4 * i + 1] = y;
			tmpMatch[4 * i + 2] = x2;
			tmpMatch[4 * i + 3] = y2;
			validMatCnt++;
		}
	}
	if (!outMatches.matchDimension(4, validMatCnt, 1)){
		outMatches.allocate(4, validMatCnt, 1);
	}
	int tmpIdx = 0;
	for (int i = 0; i < numV; i++){
		if (tmpMatch[4 * i + 0] >= 0){
			memcpy(outMatches.rowPtr(tmpIdx), tmpMatch.rowPtr(i), sizeof(int) * 4);
			tmpIdx++;
		}
	}

	return validMatCnt;
    //return 0;
}

void CPM::imDaisy(FImage& img, FImage& outFtImg)
{
	FImage imgray;
	img.desaturate(imgray);

	int w = imgray.width();
	int h = imgray.height();

	// use the version in OpenCV
	cv::Ptr<cv::xfeatures2d::DAISY> daisy =
		cv::xfeatures2d::DAISY::create(5, 3, 4, 8,
		cv::xfeatures2d::DAISY::NRM_FULL, cv::noArray(), false, false);
	cv::Mat cvImg(h, w, CV_8UC1);
	for (int i = 0; i < h; i++){
		for (int j = 0; j < w; j++){
			cvImg.at<unsigned char>(i, j) = imgray[i*w + j] * 255;
		}
	}
	cv::Mat outFeatures;
	daisy->compute(cvImg, outFeatures);
	cout<< "height: " <<outFeatures.rows<<endl;
	cout<< "width: " << outFeatures.cols<<endl;

	int itSize = outFeatures.cols;
	outFtImg.allocate(w, h, itSize);
	for (int i = 0; i < h; i++){
		for (int j = 0; j < w; j++){
			int idx = i*w + j;
			for (int k = 0; k < itSize; k++){
				outFtImg.pData[idx*itSize + k] = outFeatures.at<float>(idx, k) * 255;
			}
		}
	}
}

void CPM::imAzin(FImage& img, FImage& outFtImg)
{

	int w = img.width();
	int h = img.height();

	cv::Mat cvImg(h, w, CV_8UC3);
for (int k = 0; k<3; k++)
{
    for (int i = 0; i < h; i++)
    {
		for (int j = 0; j < w; j++)
		{
			    cvImg.at<Vec3b>(i, j)[k] = img[i*w*3 + j*3 + k] * 255;
        }
	}
}

	cv::Mat outFeatures = azin_multiPooling(cvImg);
	int itSize = outFeatures.cols;
	outFtImg.allocate(w , h , itSize);
	for (int i = 0; i < h; i++){
		for (int j = 0; j <w; j++){
			int idx = i*w + j;
			for (int k = 0; k < itSize; k++){
				outFtImg.pData[idx*itSize + k] = outFeatures.at<float>(idx, k);
			}
		}
	}

	cout<<"End of ImAzin***************************"<<endl;

}


void CPM::CrossCheck(IntImage& seeds, FImage& seedsFlow, FImage& seedsFlow2, IntImage& kLabel2, int* valid, float th)
{
	int w = kLabel2.width();
	int h = kLabel2.height();
	int numV = seeds.height();
	for (int i = 0; i < numV; i++){
		valid[i] = 1;
	}

	// cross check (1st step)
	int b = _borderWidth;
	for (int i = 0; i < numV; i++){
		float u = seedsFlow[2 * i];
		float v = seedsFlow[2 * i + 1];
		int x = seeds[2 * i];
		int y = seeds[2 * i + 1];
		int x2 = x + u;
		int y2 = y + v;
		if (x < b || x >= w - b || y < b || y >= h - b
			|| x2 < b || x2 >= w - b || y2 < b || y2 >= h - b
			|| sqrt(u*u + v*v)>_maxDisplacement){
			valid[i] = 0;
			continue;
		}

		int idx2 = kLabel2[y2*w + x2];
		float u2 = seedsFlow2[2 * idx2];
		float v2 = seedsFlow2[2 * idx2 + 1];
		float diff = sqrt((u + u2)*(u + u2) + (v + v2)*(v + v2));
		if (diff > th){
			valid[i] = 0;
		}

	}
}

float CPM::MatchCost(FImage& img1, FImage& img2, FImage* im1f, FImage* im2f, int x1, int y1, int x2, int y2)
{
	int w = im1f->width();
	int h = im1f->height();
	int ch = im1f->nchannels();
	float totalDiff;

	// fast
	x1 = ImageProcessing::EnforceRange(x1, w);
	x2 = ImageProcessing::EnforceRange(x2, w);
	y1 = ImageProcessing::EnforceRange(y1, h);
	y2 = ImageProcessing::EnforceRange(y2, h);

	float* p1 = im1f->pixPtr(y1, x1);
	float* p2 = im2f->pixPtr(y2, x2);

	totalDiff = 0;

#ifdef WITH_SSE
#undef WITH_SSE
#endif // WITH_SSE

#if false
	// SSE2
	float *_p1 = p1, *_p2 = p2;
	hu_m128 r1, r2, r3;
	int iterCnt = ch / 16;
	int idx = 0;
	int sum0 = 0;
	int sum1 = 0;
	for (idx = 0; idx < iterCnt; idx++){
		memcpy(&r1, _p1, sizeof(hu_m128));
		memcpy(&r2, _p2, sizeof(hu_m128));
		_p1 += sizeof(hu_m128);
		_p2 += sizeof(hu_m128);
		r3.mi = _mm_sad_epu8(r1.mi, r2.mi);
		sum0 += r3.m128i_u16[0];
		sum1 += r3.m128i_u16[4];
	}
	totalDiff += sum0;
	totalDiff += sum1;
	// add the left
	for (idx *= 16; idx < ch; idx++){
		totalDiff += abs(p1[idx] - p2[idx]);
	}
#else
	totalDiff = 0;
	for (int idx = 0; idx < ch; idx++){
		totalDiff += abs(p1[idx] - p2[idx]);
	}
#endif

	return totalDiff;
}

int CPM::Propogate(FImagePyramid& pyd1, FImagePyramid& pyd2, FImage* pyd1f, FImage* pyd2f, int level, float* radius, int iterCnt, IntImage* pydSeeds, IntImage& neighbors, FImage* pydSeedsFlow, float* bestCosts)
{
	int nLevels = pyd1.nlevels();
	float ratio = pyd1.ratio();

	FImage im1 = pyd1[level];
	FImage im2 = pyd2[level];
	FImage* im1f = pyd1f + level;
	FImage* im2f = pyd2f + level;
	IntImage* seeds = pydSeeds + level;
	FImage* seedsFlow = pydSeedsFlow + level;

	int w = im1.width();
	int h = im1.height();
	int ptNum = seeds->height();

	int maxNb = neighbors.width();
	int* vFlags = new int[ptNum];

	// init cost
	for (int i = 0; i < ptNum; i++){
		int x = seeds->pData[2 * i];
		int y = seeds->pData[2 * i + 1];
		float u = seedsFlow->pData[2 * i];
		float v = seedsFlow->pData[2 * i + 1];
		bestCosts[i] = MatchCost(im1, im2, im1f, im2f, x, y, x + u, y + v);
	}

	int iter = 0;
	float lastUpdateRatio = 2;
	for (iter = 0; iter < _maxIters; iter++)
	{
		int updateCount = 0;

		memset(vFlags, 0, sizeof(int)*ptNum);

		int startPos = 0, endPos = ptNum, step = 1;
		if (iter % 2 == 1){
			startPos = ptNum - 1; endPos = -1; step = -1;
		}
		for (int pos = startPos; pos != endPos; pos += step){
			bool updateFlag = false;

			int idx = pos;

			int x = seeds->pData[2 * idx];
			int y = seeds->pData[2 * idx + 1];

			int* nbIdx = neighbors.rowPtr(idx);
			// Propagation: Improve current guess by trying instead correspondences from neighbors
			for (int i = 0; i < maxNb; i++){
				if (nbIdx[i] < 0){
					break;
				}
				if (!vFlags[nbIdx[i]]){ // unvisited yet
					continue;
				}
				float tu = seedsFlow->pData[2 * nbIdx[i]];
				float tv = seedsFlow->pData[2 * nbIdx[i] + 1];
				float cu = seedsFlow->pData[2 * idx];
				float cv = seedsFlow->pData[2 * idx + 1];
				if (abs(tu - cu) < 1e-6 && abs(tv - cv) < 1e-6){
					continue;
				}
				float tc = MatchCost(im1, im2, im1f, im2f, x, y, x + tu, y + tv);
				if (tc < bestCosts[idx]){
					bestCosts[idx] = tc;
					seedsFlow->pData[2 * idx] = tu;
					seedsFlow->pData[2 * idx + 1] = tv;
					updateFlag = true;
				}
			}

			// Random search: Improve current guess by searching in boxes
			// of exponentially decreasing size around the current best guess.
			for (int mag = radius[idx] + 0.5; mag >= 1; mag /= 2) {
				/* Sampling window */
				float tu = seedsFlow->pData[2 * idx] + rand() % (2 * mag + 1) - mag;

				float tv = 0;
				if (!_isStereo){
					tv = seedsFlow->pData[2 * idx + 1] + rand() % (2 * mag + 1) - mag;
				}

				float cu = seedsFlow->pData[2 * idx];
				float cv = seedsFlow->pData[2 * idx + 1];
				if (abs(tu - cu) < 1e-6 && abs(tv - cv) < 1e-6){
					continue;
				}

				float tc = MatchCost(im1, im2, im1f, im2f, x, y, x + tu, y + tv);
				if (tc < bestCosts[idx]){
					bestCosts[idx] = tc;
					seedsFlow->pData[2 * idx] = tu;
					seedsFlow->pData[2 * idx + 1] = tv;
					updateFlag = true;
				}
			}
			vFlags[idx] = 1;
			//ShowSuperPixelFlow(spt, img1, bestU, bestV, ptNum);

			if (updateFlag){
				updateCount++;
			}
		}
		//printf("iter %d: %f [s]\n", iter, t.toc());

		float updateRatio = float(updateCount) / ptNum;
		//printf("Update ratio: %f\n", updateRatio);
		if (updateRatio < _stopIterRatio || lastUpdateRatio - updateRatio < 0.01){
			iter++;
			break;
		}
		lastUpdateRatio = updateRatio;
	}

	delete[] vFlags;

	return iter;
}

void CPM::PyramidRandomSearch(FImagePyramid& pyd1, FImagePyramid& pyd2, FImage* im1f, FImage* im2f, IntImage* pydSeeds, IntImage& neighbors, FImage* pydSeedsFlow)
{
	int nLevels = pyd1.nlevels();
	float ratio = pyd1.ratio();

	FImage rawImg1 = pyd1[0];
	FImage rawImg2 = pyd2[0];
	srand(0);

	int w = rawImg1.width();
	int h = rawImg1.height();
	int numV = pydSeeds[0].height();

	float* bestCosts = new float[numV];
	float* searchRadius = new float[numV];

	// random Initialization on coarsest level
	int initR = _maxDisplacement * pow(ratio, nLevels - 1) + 0.5;
	for (int i = 0; i < numV; i++){
		pydSeedsFlow[nLevels - 1][2 * i] = rand() % (2 * initR + 1) - initR;
		if (_isStereo){
			pydSeedsFlow[nLevels - 1][2 * i + 1] = 0;
		}else{
			pydSeedsFlow[nLevels - 1][2 * i + 1] = rand() % (2 * initR + 1) - initR;
		}
	}

	// set the radius of coarsest level
	for (int i = 0; i < numV; i++){
		searchRadius[i] = initR;
	}

	int* iterCnts = new int[nLevels];
	for (int i = 0; i < nLevels; i++){
		iterCnts[i] = _maxIters;
	}

	for (int l = nLevels - 1; l >= 0; l--){ // coarse-to-fine
		int iCnt = Propogate(pyd1, pyd2, im1f, im2f, l, searchRadius, iterCnts[l], pydSeeds, neighbors, pydSeedsFlow, bestCosts);
		if (l > 0){
			UpdateSearchRadius(neighbors, pydSeedsFlow, l, searchRadius);

			// scale the radius accordingly
			int maxR = __min(32, _maxDisplacement * pow(ratio, l) + 0.5);
			for (int i = 0; i < numV; i++){
				searchRadius[i] = __max(__min(searchRadius[i], maxR), 1);
				searchRadius[i] *= (1. / _pydRatio);
			}

			pydSeedsFlow[l - 1].copyData(pydSeedsFlow[l]);
			pydSeedsFlow[l - 1].Multiplywith(1. / ratio);
		}
	}

	delete[] searchRadius;
	delete[] bestCosts;
	delete[] iterCnts;
}

void CPM::OnePass(FImagePyramid& pyd1, FImagePyramid& pyd2, FImage* im1f, FImage* im2f, IntImage& seeds, IntImage& neighbors, FImage* pydSeedsFlow)
{
	FImage rawImg1 = pyd1[0];
	FImage rawImg2 = pyd2[0];

	int nLevels = pyd1.nlevels();
	float ratio = pyd1.ratio();

	int numV = seeds.height();

	IntImage* pydSeeds = new IntImage[nLevels];
	for (int i = 0; i < nLevels; i++){
		pydSeeds[i].allocate(2, numV);
		int sw = pyd1[i].width();
		int sh = pyd1[i].height();
		for (int n = 0; n < numV; n++){
			pydSeeds[i][2 * n] = ImageProcessing::EnforceRange(seeds[2 * n] * pow(ratio, i), sw);
			pydSeeds[i][2 * n + 1] = ImageProcessing::EnforceRange(seeds[2 * n + 1] * pow(ratio, i), sh);
		}
	}

	PyramidRandomSearch(pyd1, pyd2, im1f, im2f, pydSeeds, neighbors, pydSeedsFlow);

	// scale
	int b = _borderWidth;
	for (int i = 0; i < nLevels; i++){
		pydSeedsFlow[i].Multiplywith(pow(1. / ratio, i));
	}

	delete[] pydSeeds;
}

void CPM::UpdateSearchRadius(IntImage& neighbors, FImage* pydSeedsFlow, int level, float* outRadius)
{
	FImage* seedsFlow = pydSeedsFlow + level;
	int maxNb = neighbors.width();

	float x[32], y[32]; // for minimal circle
	assert(maxNb < 32);

	int sCnt = seedsFlow->height();
	for (int i = 0; i < sCnt; i++){
		// add itself
		x[0] = seedsFlow->pData[2 * i];
		y[0] = seedsFlow->pData[2 * i + 1];
		int nbCnt = 1;

		// add neighbors
		int* nbIdx = neighbors.rowPtr(i);
		for (int n = 0; n < maxNb; n++){
			if (nbIdx[n] < 0){
				break;
			}

			x[nbCnt] = seedsFlow->pData[2 * nbIdx[n]];
			y[nbCnt] = seedsFlow->pData[2 * nbIdx[n] + 1];
			nbCnt++;
		}

		float circleR = MinimalCircle(x, y, nbCnt);
		outRadius[i] = circleR;
	}
}

double CPM::dist(Point a, Point b)
{
	return sqrt((a.x - b.x) * (a.x - b.x) + (a.y - b.y) * (a.y - b.y));
}

// intersection between two lines
CPM::Point CPM::intersection(Point u1, Point u2, Point v1, Point v2)
{
	Point ans = u1;
	double t = ((u1.x - v1.x) * (v1.y - v2.y) - (u1.y - v1.y) * (v1.x - v2.x)) /
		((u1.x - u2.x) * (v1.y - v2.y) - (u1.y - u2.y) * (v1.x - v2.x));
	ans.x += (u2.x - u1.x) * t;
	ans.y += (u2.y - u1.y) * t;
	return ans;
}

// circle center containing a triangular
CPM::Point CPM::circumcenter(Point a, Point b, Point c)
{
	Point ua, ub, va, vb;
	ua.x = (a.x + b.x) / 2;
	ua.y = (a.y + b.y) / 2;
	ub.x = ua.x - a.y + b.y;//根据 垂直判断，两线段点积为0
	ub.y = ua.y + a.x - b.x;
	va.x = (a.x + c.x) / 2;
	va.y = (a.y + c.y) / 2;
	vb.x = va.x - a.y + c.y;
	vb.y = va.y + a.x - c.x;
	return intersection(ua, ub, va, vb);
}

float CPM::MinimalCircle(float* x, float*y, int n, float* centerX, float* centerY)
{
	static double eps = 1e-6;

	// prepare data
	Point p[20];
	assert(n < 20);
	for (int i = 0; i < n; i++){
		p[i].x = x[i];
		p[i].y = y[i];
	}
	// center and radius of the circle
	Point o;
	double r;

	int i, j, k;
	o = p[0];
	r = 0;
	for (i = 1; i < n; i++)//准备加入的点
	{
		if (dist(p[i], o) - r > eps)//如果第i点在 i-1前最小圆外面
		{
			o = p[i];//另定圆心
			r = 0;//另定半径

			for (j = 0; j < i; j++)//循环再确定半径
			{
				if (dist(p[j], o) - r > eps)
				{
					o.x = (p[i].x + p[j].x) / 2.0;
					o.y = (p[i].y + p[j].y) / 2.0;

					r = dist(o, p[j]);

					for (k = 0; k < j; k++)
					{
						if (dist(o, p[k]) - r > eps)//如果j前面有点不符和 i与j确定的圆，则更新
						{
							o = circumcenter(p[i], p[j], p[k]);
							r = dist(o, p[k]);
						}
					}//循环不超过3层，因为一个圆最多3个点可以确定
				}
			}
		}
	}

	if (centerX){
		*centerX = o.x;
	}
	if (centerY){
		*centerY = o.y;
	}
	return r;
}

