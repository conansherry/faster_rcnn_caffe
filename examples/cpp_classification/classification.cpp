#include <caffe/caffe.hpp>
#include <caffe/util/misc.hpp>
#include <opencv2/opencv.hpp>
#include <algorithm>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>

using namespace caffe;  // NOLINT(build/namespaces)
using std::string;

/* Pair (label, confidence) representing a prediction. */
typedef std::pair<string, float> Prediction;

class Classifier {
 public:
  Classifier(const string& model_file,
             const string& trained_file,
             const string& mean_file,
             const string& label_file);

  std::vector<Prediction> Classify(const cv::Mat& img, int N = 5);

 private:
  void SetMean(const string& mean_file);

  std::vector<float> Predict(const cv::Mat& img);

  void WrapInputLayer(std::vector<cv::Mat>* input_channels);

  void Preprocess(const cv::Mat& img,
                  std::vector<cv::Mat>* input_channels);

 private:
  shared_ptr<Net<float> > net_;
  cv::Size input_geometry_;
  int num_channels_;
  cv::Mat mean_;
  std::vector<string> labels_;
};

Classifier::Classifier(const string& model_file,
                       const string& trained_file,
                       const string& mean_file,
                       const string& label_file) {
#ifdef CPU_ONLY
  Caffe::set_mode(Caffe::CPU);
#else
  Caffe::set_mode(Caffe::GPU);
#endif

  /* Load the network. */
  net_.reset(new Net<float>(model_file, TEST));
  net_->CopyTrainedLayersFrom(trained_file);

  CHECK_EQ(net_->num_inputs(), 1) << "Network should have exactly one input.";
  CHECK_EQ(net_->num_outputs(), 1) << "Network should have exactly one output.";

  Blob<float>* input_layer = net_->input_blobs()[0];
  num_channels_ = input_layer->channels();
  CHECK(num_channels_ == 3 || num_channels_ == 1)
    << "Input layer should have 1 or 3 channels.";
  input_geometry_ = cv::Size(input_layer->width(), input_layer->height());

  /* Load the binaryproto mean file. */
  SetMean(mean_file);

  /* Load labels. */
  std::ifstream labels(label_file.c_str());
  CHECK(labels) << "Unable to open labels file " << label_file;
  string line;
  while (std::getline(labels, line))
    labels_.push_back(string(line));

  Blob<float>* output_layer = net_->output_blobs()[0];
  CHECK_EQ(labels_.size(), output_layer->channels())
    << "Number of labels is different from the output layer dimension.";
}

static bool PairCompare(const std::pair<float, int>& lhs,
                        const std::pair<float, int>& rhs) {
  return lhs.first > rhs.first;
}

/* Return the indices of the top N values of vector v. */
static std::vector<int> Argmax(const std::vector<float>& v, int N) {
  std::vector<std::pair<float, int> > pairs;
  for (size_t i = 0; i < v.size(); ++i)
    pairs.push_back(std::make_pair(v[i], static_cast<int>(i)));
  std::partial_sort(pairs.begin(), pairs.begin() + N, pairs.end(), PairCompare);

  std::vector<int> result;
  for (int i = 0; i < N; ++i)
    result.push_back(pairs[i].second);
  return result;
}

/* Return the top N predictions. */
std::vector<Prediction> Classifier::Classify(const cv::Mat& img, int N) {
  std::vector<float> output = Predict(img);

  N = std::min<int>(labels_.size(), N);
  std::vector<int> maxN = Argmax(output, N);
  std::vector<Prediction> predictions;
  for (int i = 0; i < N; ++i) {
    int idx = maxN[i];
    predictions.push_back(std::make_pair(labels_[idx], output[idx]));
  }

  return predictions;
}

/* Load the mean file in binaryproto format. */
void Classifier::SetMean(const string& mean_file) {
  BlobProto blob_proto;
  ReadProtoFromBinaryFileOrDie(mean_file.c_str(), &blob_proto);

  /* Convert from BlobProto to Blob<float> */
  Blob<float> mean_blob;
  mean_blob.FromProto(blob_proto);
  CHECK_EQ(mean_blob.channels(), num_channels_)
    << "Number of channels of mean file doesn't match input layer.";

  /* The format of the mean file is planar 32-bit float BGR or grayscale. */
  std::vector<cv::Mat> channels;
  float* data = mean_blob.mutable_cpu_data();
  for (int i = 0; i < num_channels_; ++i) {
    /* Extract an individual channel. */
    cv::Mat channel(mean_blob.height(), mean_blob.width(), CV_32FC1, data);
    channels.push_back(channel);
    data += mean_blob.height() * mean_blob.width();
  }

  /* Merge the separate channels into a single image. */
  cv::Mat mean;
  cv::merge(channels, mean);

  /* Compute the global mean pixel value and create a mean image
   * filled with this value. */
  cv::Scalar channel_mean = cv::mean(mean);
  mean_ = cv::Mat(input_geometry_, mean.type(), channel_mean);
}

std::vector<float> Classifier::Predict(const cv::Mat& img) {
  Blob<float>* input_layer = net_->input_blobs()[0];
  input_layer->Reshape(1, num_channels_,
                       input_geometry_.height, input_geometry_.width);
  /* Forward dimension change to all layers. */
  net_->Reshape();

  std::vector<cv::Mat> input_channels;
  WrapInputLayer(&input_channels);

  Preprocess(img, &input_channels);

  net_->Forward();

  /* Copy the output layer to a std::vector */
  Blob<float>* output_layer = net_->output_blobs()[0];
  const float* begin = output_layer->cpu_data();
  const float* end = begin + output_layer->channels();
  return std::vector<float>(begin, end);
}

/* Wrap the input layer of the network in separate cv::Mat objects
 * (one per channel). This way we save one memcpy operation and we
 * don't need to rely on cudaMemcpy2D. The last preprocessing
 * operation will write the separate channels directly to the input
 * layer. */
void Classifier::WrapInputLayer(std::vector<cv::Mat>* input_channels) {
  Blob<float>* input_layer = net_->input_blobs()[0];

  int width = input_layer->width();
  int height = input_layer->height();
  float* input_data = input_layer->mutable_cpu_data();
  for (int i = 0; i < input_layer->channels(); ++i) {
    cv::Mat channel(height, width, CV_32FC1, input_data);
    input_channels->push_back(channel);
    input_data += width * height;
  }
}

void Classifier::Preprocess(const cv::Mat& img,
                            std::vector<cv::Mat>* input_channels) {
  /* Convert the input image to the input image format of the network. */
  cv::Mat sample;
  if (img.channels() == 3 && num_channels_ == 1)
    cv::cvtColor(img, sample, cv::COLOR_BGR2GRAY);
  else if (img.channels() == 4 && num_channels_ == 1)
    cv::cvtColor(img, sample, cv::COLOR_BGRA2GRAY);
  else if (img.channels() == 4 && num_channels_ == 3)
    cv::cvtColor(img, sample, cv::COLOR_BGRA2BGR);
  else if (img.channels() == 1 && num_channels_ == 3)
    cv::cvtColor(img, sample, cv::COLOR_GRAY2BGR);
  else
    sample = img;

  cv::Mat sample_resized;
  if (sample.size() != input_geometry_)
    cv::resize(sample, sample_resized, input_geometry_);
  else
    sample_resized = sample;

  cv::Mat sample_float;
  if (num_channels_ == 3)
    sample_resized.convertTo(sample_float, CV_32FC3);
  else
    sample_resized.convertTo(sample_float, CV_32FC1);

  cv::Mat sample_normalized;
  cv::subtract(sample_float, mean_, sample_normalized);

  /* This operation will write the separate BGR planes directly to the
   * input layer of the network because it is wrapped by the cv::Mat
   * objects in input_channels. */
  cv::split(sample_normalized, *input_channels);

  CHECK(reinterpret_cast<float*>(input_channels->at(0).data)
        == net_->input_blobs()[0]->cpu_data())
    << "Input channels are not wrapping the input layer of the network.";
}

//template<typename Dtype>
//class Box {
//public:
//    Box() : x1(0), y1(0), x2(0), y2(0) {}
//    Box(Dtype x1_, Dtype y1_, Dtype x2_, Dtype y2_)
//        : x1(x1_), y1(y1_), x2(x2_), y2(y2_) {}
//    Box(const Box& box, Dtype shift_x, Dtype shift_y)
//        : x1(box.x1 + shift_x), y1(box.y1 + shift_y),
//        x2(box.x2 + shift_x), y2(box.y2 + shift_y) {}
//
//    Dtype x1, y1, x2, y2;
//};
//
//template <typename Dtype>
//void clip_boxes(std::vector<Box<Dtype> >& boxes, int im_rows, int im_cols)
//{
//    for (int i = 0; i < boxes.size(); i++)
//    {
//        boxes[i].x1 = std::max(std::min(boxes[i].x1, (Dtype)im_cols - 1), (Dtype)0);
//        boxes[i].y1 = std::max(std::min(boxes[i].y1, (Dtype)im_rows - 1), (Dtype)0);
//        boxes[i].x2 = std::max(std::min(boxes[i].x2, (Dtype)im_cols - 1), (Dtype)0);
//        boxes[i].y2 = std::max(std::min(boxes[i].y2, (Dtype)im_rows - 1), (Dtype)0);
//    }
//}
//
//template <typename Dtype>
//bool myfunction1(const std::pair<Dtype, size_t>& i, const std::pair<Dtype, size_t>& j)
//{
//    return (i.first<j.first);
//}
//
//template <typename Dtype>
//bool myfunction2(const std::pair<Dtype, size_t>& i, const std::pair<Dtype, size_t>& j)
//{
//    return (i.first>j.first);
//}
//
//template <typename Dtype>
//std::vector<int> sort_ind(const std::vector<Dtype> &scores, bool increasing)
//{
//    std::vector<std::pair<Dtype, size_t> > vp;
//    vp.reserve(scores.size());
//    for (size_t i = 0; i != scores.size(); i++)
//        vp.push_back(std::make_pair(scores[i], i));
//
//    if (increasing)
//        std::sort(vp.begin(), vp.end(), myfunction1<Dtype>);
//    else
//        std::sort(vp.begin(), vp.end(), myfunction2<Dtype>);
//    std::vector<int> ind;
//    for (size_t i = 0; i != vp.size(); i++) {
//        ind.push_back(vp[i].second);
//    }
//    return ind;
//}
//
//template <typename Dtype>
//std::vector<int> nms(const std::vector<Box<Dtype> >& boxes,
//    const std::vector<Dtype>& scores, const Dtype& nms_thresh, float _th)
//{
//    std::vector<int> ind = sort_ind(scores, false);
//    std::vector<bool> suppressed(ind.size(), false);
//    std::vector<int> keep_ind;
//
//    for (int _i = 0; _i < boxes.size() * _th; _i++)
//    {
//        int i = ind[_i];
//        if (suppressed[i])
//            continue;
//
//        keep_ind.push_back(i);
//
//        Dtype ix1 = boxes[i].x1;
//        Dtype iy1 = boxes[i].y1;
//        Dtype ix2 = boxes[i].x2;
//        Dtype iy2 = boxes[i].y2;
//        Dtype iarea = (ix2 - ix1 + 1)*(iy2 - iy1 + 1);
//
//        for (int _j = _i + 1; _j < boxes.size() * _th; _j++)
//        {
//            int j = ind[_j];
//            if (suppressed[j])
//                continue;
//
//            Dtype xx1 = std::max(ix1, boxes[j].x1);
//            Dtype yy1 = std::max(iy1, boxes[j].y1);
//            Dtype xx2 = std::min(ix2, boxes[j].x2);
//            Dtype yy2 = std::min(iy2, boxes[j].y2);
//            Dtype w = std::max(Dtype(0.0), xx2 - xx1 + 1);
//            Dtype h = std::max(Dtype(0.0), yy2 - yy1 + 1);
//
//            Dtype inter = w * h;
//            Dtype jarea = (boxes[j].x2 - boxes[j].x1 + 1) * (boxes[j].y2 - boxes[j].y1 + 1);
//            Dtype ovr = inter / (iarea + jarea - inter);
//            if (ovr >= nms_thresh)
//                suppressed[j] = true;
//        }
//    }
//
//    return keep_ind;
//}
//
//template <typename T>
//std::vector<T> keep(std::vector<T> items, std::vector<int> kept_ind)
//{
//    std::vector<T> newitems;
//    for (int i = 0; i< std::min(kept_ind.size(), items.size()); i++)
//        newitems.push_back(items[kept_ind[i]]);
//    return newitems;
//}
//
struct DetectBox
{
    DetectBox(const cv::Rect& _rect = cv::Rect(0, 0, 0, 0),
        float _score = -1,
        size_t _cls_idx = 9999)
    {
        rect = _rect;
        score = _score;
        class_index = _cls_idx;
    }
    cv::Rect rect;
    float score;
    size_t class_index;
};

int main(int argc, char** argv)
{
    std::shared_ptr<Net<float> > net;
    net.reset(new Net<float>("F:/workspace/20180122/linux/test.prototxt", TEST));
    net->CopyTrainedLayersFrom("F:/workspace/20180122/linux/test.caffemodel");

    Caffe::set_mode(Caffe::GPU);
    Caffe::SetDevice(0);

    cv::VideoCapture cap(0);
    while (true)
    {
        cv::Mat image;
        cap >> image;
        cv::Mat display = image.clone();
        //image = cv::imread("G:/temp/input.png");

        std::vector<DetectBox> _res;

        // resize
        float scale = 1.0;
        if (image.cols < image.rows)
        {
            scale = 600. / image.cols;
            cv::resize(image, image, cv::Size(0, 0), scale, scale, cv::INTER_LINEAR);
        }
        else
        {
            scale = 600. / image.rows;
            cv::resize(image, image, cv::Size(0, 0), scale, scale, cv::INTER_LINEAR);
        }
        image.convertTo(image, CV_32F);
        image = image - cv::Scalar(102.9801, 115.9465, 122.7717);

        const std::vector<Blob<float>* > blobs = net->input_blobs();
        blobs[0]->Reshape(1, 3, image.rows, image.cols);
        blobs[1]->Reshape({ 1, 3 });
        net->Reshape();
       
        std::vector<cv::Mat> input_channels;
        int width = blobs[0]->width();
        int height = blobs[0]->height();
        float* input_data = blobs[0]->mutable_cpu_data();
        for (int i = 0; i < blobs[0]->channels(); ++i) {
            cv::Mat channel(height, width, CV_32FC1, input_data);
            input_channels.push_back(channel);
            input_data += width * height;
        }

        cv::split(image, input_channels);

        //std::ifstream fin("G:/temp/blob.txt");
        //for (int i = 0; i < blobs[0]->count(); i++)
        //{
        //    fin >> blobs[0]->mutable_cpu_data()[i];
        //}

        std::cout << blobs[0]->mutable_cpu_data()[0] << " " << blobs[0]->mutable_cpu_data()[1] << std::endl;

        blobs[1]->mutable_cpu_data()[0] = image.rows;
        blobs[1]->mutable_cpu_data()[1] = image.cols;
        blobs[1]->mutable_cpu_data()[2] = scale;

        int64 t0 = cv::getTickCount();
        net->Forward();

        //for (auto &name : net->blob_names())
        //{
        //    Blob<float>* blob = net->blob_by_name(name).get();
        //    std::ofstream fout("G:/temp/" + name + ".out");
        //    for (int i = 0; i < blob->count(); i++)
        //    {
        //        fout << blob->cpu_data()[i] << " ";
        //    }
        //}

        std::cout << "cost " << (cv::getTickCount() - t0) / cv::getTickFrequency() * 1000 << std::endl;

        Blob<float>* rois_blob = net->blob_by_name("rois").get();
        Blob<float>* box_deltas_blob = net->blob_by_name("bbox_pred").get();
        Blob<float>* score_blob = net->blob_by_name("cls_prob").get();

        std::cout << rois_blob->shape_string() << std::endl;

        int roi_num = score_blob->num();
        int class_num = score_blob->channels();

        assert(roi_num == rois_blob->num());
        assert(roi_num == score_blob->num());
        assert(class_num * 4 == box_deltas_blob->channels());
        assert(roi_num == score_blob->num());
        assert(class_num == score_blob->channels());

        for (int class_idx = 1; class_idx < class_num; ++class_idx)
        {
            std::vector< Box<float> > class_boxes;
            std::vector<float> class_scores;
            for (int roi_idx = 0; roi_idx < roi_num; ++roi_idx)
            {
                float x1 = rois_blob->cpu_data()[rois_blob->offset(roi_idx, 1, 0, 0)] / scale;
                float y1 = rois_blob->cpu_data()[rois_blob->offset(roi_idx, 2, 0, 0)] / scale;
                float x2 = rois_blob->cpu_data()[rois_blob->offset(roi_idx, 3, 0, 0)] / scale;
                float y2 = rois_blob->cpu_data()[rois_blob->offset(roi_idx, 4, 0, 0)] / scale;

                float orig_w = x2 - x1 + 1.0;
                float orig_h = y2 - y1 + 1.0;
                float orig_cx = x1 + (orig_w) * 0.5;
                float orig_cy = y1 + (orig_h) * 0.5;

                float dx = box_deltas_blob->cpu_data()[box_deltas_blob->offset(roi_idx, class_idx * 4, 0, 0)];
                float dy = box_deltas_blob->cpu_data()[box_deltas_blob->offset(roi_idx, class_idx * 4 + 1, 0, 0)];
                float dw = box_deltas_blob->cpu_data()[box_deltas_blob->offset(roi_idx, class_idx * 4 + 2, 0, 0)];
                float dh = box_deltas_blob->cpu_data()[box_deltas_blob->offset(roi_idx, class_idx * 4 + 3, 0, 0)];

                float pred_cx = orig_cx + orig_w * dx;
                float pred_cy = orig_cy + orig_h * dy;
                float pred_w = orig_w * std::exp(dw);
                float pred_h = orig_h * std::exp(dh);

                class_boxes.push_back(Box<float>(pred_cx - 0.5 * (pred_w), pred_cy - 0.5 * (pred_h), pred_cx + 0.5 * (pred_w), pred_cy + 0.5 * (pred_h)));
                class_scores.push_back(score_blob->cpu_data()[score_blob->offset(roi_idx, class_idx, 0, 0)]);
            } // for roi_idx

            clip_boxes(class_boxes, (int)image.rows, (int)image.cols);

            std::vector<int> keep_index = nms(class_boxes, class_scores, 0.01f);
            class_boxes = keep(class_boxes, keep_index);
            class_scores = keep(class_scores, keep_index);

            for (int idx = 0; idx < class_scores.size(); ++idx)
            {
                if (class_scores.at(idx) < 0.9) { continue; }
                int tmp_x = class_boxes.at(idx).x1;
                int tmp_y = class_boxes.at(idx).y1;
                int tmp_w = class_boxes.at(idx).x2 - tmp_x + 1;
                int tmp_h = class_boxes.at(idx).y2 - tmp_y + 1;

                _res.push_back(DetectBox(cv::Rect(tmp_x, tmp_y, tmp_w, tmp_h), class_scores.at(idx), class_idx));

            }
        }

        for (auto &det_box : _res)
        {
            std::cout << det_box.rect << " " << det_box.score << std::endl;
        }

        cv::imshow("image", display);
        cv::waitKey(20);
    }
}
