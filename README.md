# Medical-VQA

VQA systems or visual question/answer systems aims to answer specific questions, on images that given as inputs. This is a relatively recent research topic,
for which different datasets and approaches have already been exploited in the
literature.

In this Project, we are interested in VQA systems in the medical field, specifically
in the performance of deep learning algorithms for this problem.
First, we determined the state of the art of recent work. Since there are different
datasets, we studied them and highlighted their characteristics to choose the
one that is best suited to the approaches we explored. Then, we proposed an
architecture based on transformers, using BioBERT for question processing and
ViT for images. This architecture was implemented, evaluated, and compared
to chosen baselines. Finally, the best architecture was deployed as an API and
tested on images from the medical field.
