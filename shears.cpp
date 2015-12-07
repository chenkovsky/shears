// By default, SRILM defines a function called zopen.
//
// However, on Mac OS X (and possibly other BSDs),
// <stdio.h> already defines a zopen function.
//
// To resolve this conflict, SRILM checks to see if HAVE_ZOPEN is defined.
// If it is, SRILM will rename its zopen function as my_zopen.
//
// So, before importing any SRILM headers,
// it is important to define HAVE_ZOPEN if we are on an Apple OS:
//

#ifdef __APPLE__
#define HAVE_ZOPEN
#endif

#include <File.h>
#include <Ngram.h>
#define MAX_ORDER 10
using namespace std;
static bool _debug = false;

struct EntropyInfo {
    Prob deltaEntropy;
    size_t gram_id;
    bool operator<(const EntropyInfo &r) const { return deltaEntropy < r.deltaEntropy; }
};


class NgramPlus : public Ngram {
    NgramPlus(const Ngram& ngram);
public:
    NgramPlus(Vocab& vocab): Ngram(vocab){
    }
    void shear(bool is_cut, size_t *ncut, LM *historyLM = NULL) {
        /*
         * Hack alert: allocate the context buffer for NgramBOsIter, but leave
         * room for a word id to be prepended.
         */
        makeArray(VocabIndex, wordPlusContext, order + 2);
        VocabIndex *context = &wordPlusContext[1];
        for (unsigned i = order - 1; i > 0 && i >= 1; i--) { //i>=1, won't prune vocab
            size_t cur_ncut = ncut[i];
            if (cur_ncut == 0) {
                continue; //no need to cut
            }
            size_t gram_num = this->numNgrams(i + 1);
            bool* is_pruned = (bool*)malloc(sizeof(bool)*gram_num);
            memset(is_pruned, 0, sizeof(bool)*gram_num);
            //VocabIndex *indices = (VocabIndex *)malloc(sizeof(VocabIndex) * (i + 2) * gram_num); //将word连同contex存在数组中
            EntropyInfo *entropys = (EntropyInfo *)malloc(sizeof(EntropyInfo) * gram_num);
            size_t index = 0;
            size_t gram_id = 0;
            BOnode *node;
            NgramBOsIter iter1(*this, context, i);

            while ((node = iter1.next())) {
                LogP bow = node->bow;   /* old backoff weight, BOW(h) */
                double numerator, denominator;

                /* 
                 * Compute numerator and denominator of the backoff weight,
                 * so that we can quickly compute the BOW adjustment due to
                 * leaving out one prob.
                 */
                if (!computeBOW(node, context, i, numerator, denominator)) {
                    continue;
                }

                /*
                 * Compute the marginal probability of the context, P(h)
                 * If a historyLM was given (e.g., for KN smoothed LMs), use it,
                 * otherwise use the LM to be pruned.
                 */
                LogP cProb = historyLM == NULL ?
                    contextProb(context, i) :
                    historyLM->contextProb(context, i);

                NgramProbsIter piter(*node);
                VocabIndex word;
                LogP *ngramProb;

                while ((ngramProb = piter.next(word))) {
                    wordPlusContext[0] = word;
                    if (findBOW(wordPlusContext)) {//有以当前节点为prefix的高阶gram，当前的节点不能裁剪
                        gram_id++;
            			continue;
            		}
                    /*
                     * lower-order estimate for ngramProb, P(w|h')
                     */
                    LogP backoffProb = wordProbBO(word, context, i - 1);

                    /*
                     * Compute BOW after removing ngram, BOW'(h)
                     */
                    LogP newBOW =
                        ProbToLogP(numerator + LogPtoProb(*ngramProb)) -
                        ProbToLogP(denominator + LogPtoProb(backoffProb));

                    /*
                     * Compute change in entropy due to removal of ngram
                     * deltaH = - P(H) x
                     *  {P(W | H) [log P(w|h') + log BOW'(h) - log P(w|h)] +
                     *   (1 - \sum_{v,h ngrams} P(v|h)) [log BOW'(h) - log BOW(h)]}
                     *
                     * (1-\sum_{v,h ngrams}) is the probability mass left over from
                     * ngrams of the current order, and is the same as the 
                     * numerator in BOW(h).
                     */
                    LogP deltaProb = backoffProb + newBOW - *ngramProb;
                    Prob deltaEntropy = -LogPtoProb(cProb) *
                        (LogPtoProb(*ngramProb) * deltaProb +
                         numerator * (newBOW - bow));
                    if (_debug) {
                        cerr << "CONTEXT " << (vocab.use(), context)
                            << " WORD " << vocab.getWord(word)
                            << " CONTEXTPROB " << cProb
                            << " OLDPROB " << *ngramProb
                            << " NEWPROB " << (backoffProb + newBOW)
                            << " OLDBOW " << bow
                            << " NEWBOW " << newBOW
                            << " DELTA-H " << deltaEntropy
                            << " DELTA-LOGP " << deltaProb
                            << " PA " << numerator
                            << " PB " << denominator
                            << " PH_W " << backoffProb
                            << endl;
                    }
                    //copy delta entroy info into array
                    entropys[index].deltaEntropy = deltaEntropy;
                    entropys[index].gram_id = gram_id;
                    index+=1;
                    gram_id++;
                    assert(gram_id <= gram_num);
                }
            }
            //sorting....
            if (index < cur_ncut) {
                cur_ncut = index;
            }
            nth_element(entropys, entropys + cur_ncut-1, entropys + index);
            
            //标记为删除。
            for (size_t i = 0; i < cur_ncut; i++) {
                //gramset.get(entropys[i].gram, wordPlusContext);
                is_pruned[entropys[i].gram_id] = true;
                //removeProb(wordPlusContext[0], context);
                //if (contexts.numEntries(context) == 0) {
                //    removeBOW(context);
                //}
            }
            //真正进行删除操作

            NgramBOsIter iter2(*this, context, i);
            gram_id = 0;
            while ((node = iter2.next())) {
                double numerator, denominator;
                if (!computeBOW(node, context, i, numerator, denominator)) {
                    continue;
                }
                NgramProbsIter piter(*node);
                VocabIndex word;
                LogP *ngramProb;
                while ((ngramProb = piter.next(word))) {
                    wordPlusContext[0] = word;
                    if (findBOW(wordPlusContext)) {//有以当前节点为prefix的高阶gram，当前的节点不能裁剪
                        gram_id++;
            			continue;
            		}
                    if (is_pruned[gram_id]) {
                        removeProb(wordPlusContext[0], context);
                        if (contexts.numEntries(context) == 0) {
                            removeBOW(context);
                        }
                    }
                    gram_id++;
                }
            }

            free(entropys);
            free(is_pruned);
        }

        recomputeBOWs();
    }
};

#include "docopt.h"
static const char USAGE[] =
R"(Prune language model.
This program uses entropy - based method to prune the size of back-off\
language model 'src_model' to a specific size and write to 'dst_model'.
Note that we do not ensure that during pruning process,  exactly the\
the given number of items are cut or reserved, because some items may\
contains high level children, so could not be cut.
param <count> format example: 1=100 2=3000 3=4000
Usage:
    lmprune entropy (reserve|cut) [--debug] [--important=important_ngram] <src_model> <dst_model> <count>...
    lmprune (-h | --help)
    lmprune --version

Options:
    -h --help     Show this screen.
    --version     Show version.
)";

int main(int argc, char *argv[]) {
    map<string, docopt::value> args =
        docopt::docopt(USAGE, { argv + 1, argv + argc },
                       true, // show help if requested
                       "lmprune 1.0");
    auto entropy_method_ = args.find("entropy");
    assert(entropy_method_ != args.end() && entropy_method_->second.isBool());
    assert(entropy_method_->second.asBool()&&
           "currently only support entropy method");
    auto is_reserve_ = args.find("reserve");
    auto is_cut_ = args.find("cut");
    assert(is_reserve_ != args.end()&& is_reserve_->second.isBool());
    assert(is_cut_ != args.end()&& is_cut_->second.isBool());
    bool is_reserve = is_reserve_->second.asBool();
    bool is_cut = is_cut_->second.asBool();
    assert(is_reserve != is_cut);
    auto is_debug_ = args.find("--debug");
    assert(is_debug_ != args.end() && is_debug_->second.isBool());
    _debug = is_debug_->second.asBool();

    auto src_model_ = args.find("<src_model>");
    auto dst_model_ = args.find("<dst_model>");
    auto important_ = args.find("--important");
    assert(src_model_ != args.end() && src_model_->second.isString());
    assert(dst_model_ != args.end()&& dst_model_->second.isString());
    const string &src_model = src_model_->second.asString();
    const string &dst_model = dst_model_->second.asString();
    auto ncut_str_ = args.find("<count>");
    assert(ncut_str_ != args.end()&& ncut_str_->second.isStringList());
    const vector<string> &ncut_str = ncut_str_->second.asStringList();

    size_t ncut[MAX_ORDER];
    memset(ncut, -1, sizeof(ncut));
    uint32_t max_order = 0;
    for (auto it = ncut_str.begin(); it != ncut_str.end(); ++it){
        size_t eq_offset = it->find_first_of("=");
        if (eq_offset == string::npos){
            cerr << "<count> format is illegal. it should be order=num" << endl;
            exit(1);
        }
        uint32_t cur_order = atoi(it->c_str());
        max_order = max(max_order, cur_order);
        assert(cur_order < MAX_ORDER);
        assert(cur_order != 1 && "cannot prune unigram");
        size_t num = atol(it->c_str()+ eq_offset + 1);
        ncut[cur_order-1] = num;
    }

    File file(src_model.c_str(), "r");
    Vocab *vocab = new Vocab;
    vocab->unkIsWord() = false;
    vocab->toLower() = false;

    NgramPlus *ngramLM = new NgramPlus(*vocab);
    if (!ngramLM->read(file, false)) {
        cerr << "format error in lm file " << src_model << endl;
        exit(1);
    }
    ngramLM->shear(is_cut, ncut);
    File dst(dst_model.c_str(), "w");
    ngramLM->write(dst);
    delete ngramLM;
    delete vocab;
    return 0;
}

