/*
 * Copyright (C) 2007 by
 * 
 * 	Xuan-Hieu Phan
 *	hieuxuan@ecei.tohoku.ac.jp or pxhieu@gmail.com
 * 	Graduate School of Information Sciences
 * 	Tohoku University
 *
 * GibbsLDA++ is a free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published
 * by the Free Software Foundation; either version 2 of the License,
 * or (at your option) any later version.
 *
 * GibbsLDA++ is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with GibbsLDA++; if not, write to the Free Software Foundation,
 * Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA.
 */

/* 
 * References:
 * + The Java code of Gregor Heinrich (gregor@arbylon.net)
 *   http://www.arbylon.net/projects/LdaGibbsSampler.java
 * + "Parameter estimation for text analysis" by Gregor Heinrich
 *   http://www.arbylon.net/publications/text-est.pdf
 */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "constants.h"
#include "strtokenizer.h"
#include "utils.h"
#include "dataset.h"
#include "model.h"

using namespace std;

model::~model() {

  if (p) {
    for(int a = 0; a < A; a++){
      if(p[a]){
	delete p[a];
      }
    }
  }

    if (ptrndata) {
      delete ptrndata;
    }

    if (z) {
      for (int m = 0; m < M; m++) {
	if (z[m]) {
	  delete z[m];
	}
      }
    }

    if (x) {
      for (int m = 0; m < M; m++) {
	if (x[m]) {
	  delete x[m];
	}
      }
    }
    
    if (nw) {
      for (int w = 0; w < V; w++) {
	if (nw[w]) {
	  delete nw[w];
	}
      }
    }
    
    if (na) {
      for (int x = 0; x < A; x++) {
	if (na[x]) {
	  delete na[x];
	}
      }
    } 
    
    if (nwsum) {
      delete nwsum;
    }   
     
    if (nasum) {
      delete nasum;
    }

    if (theta) {
      for (int m = 0; m < A; m++) {
	if (theta[m]) {
	  delete theta[m];
	}
      }
    }
    
    if (phi) {
      for (int k = 0; k < K; k++) {
	if (phi[k]) {
	  delete phi[k];
	}
      }
    }
  }

  void model::set_default_values() {
    wordmapfile = "wordmap.txt";
    authormapfile="authormap.txt";
    trainlogfile = "trainlog.txt";
    tassign_suffix = ".tassign";
    //tassign_for_author_suffix = ".tassign_for_author";
    theta_suffix = ".theta";
    phi_suffix = ".phi";
    others_suffix = ".others";
    twords_suffix = ".twords";
    tauthors_suffix = ".tauthors";

    dir = "./";
    dfile = "trndocs.dat";
    model_name = "model-final";    
    model_status = MODEL_STATUS_UNKNOWN;
    
    ptrndata = NULL;
    
    M = 0;
    V = 0;
    K = 100;
    A = 0;
    alpha = 50.0 / K;
    beta = 0.1;
    niters = 2000;
    liter = 0;
    savestep = 200;    
    twords = 0;
    tauthors = 0;
    withrawstrs = 0;
    
    p = NULL;
    z = NULL;
    x = NULL;
    nw = NULL;
    nd = NULL;
    na = NULL;
    nasum = NULL;
    nwsum = NULL;
    ndsum = NULL;
    theta = NULL;
    phi = NULL;
    
  }

  int model::parse_args(int argc, char ** argv) {
    return utils::parse_args(argc, argv, this);
  }

  int model::init(int argc, char ** argv) {
    // call parse_args
    if (parse_args(argc, argv)) {
      return 1;
    }
    if (model_status == MODEL_STATUS_EST) {
      // estimating the model from scratc
      if (init_est()) {
      }
    }
    return 0;
  }

  int model::save_model(string model_name) {

    if (save_model_tassign(dir + model_name + tassign_suffix)) {
      return 1;
    }
    /*    
    if (save_model_tassign_for_author(dir + model_name + tassign_for_author_suffix)){
      return 1;
    }
    */
    if (save_model_others(dir + model_name + others_suffix)) {
      return 1;
    }
    
    if (save_model_theta(dir + model_name + theta_suffix)) {
      return 1;
    }
    
    if (save_model_phi(dir + model_name + phi_suffix)) {
      return 1;
    }
    
    if (twords > 0) {
      if (save_model_twords(dir + model_name + twords_suffix)) {
	return 1;
      }
    }

    if (tauthors > 0)
      {
      if (save_model_tauthors(dir + model_name + tauthors_suffix)){
	return 1;
	}
      }
    
    return 0;
  }

  int model::save_model_tassign(string filename) {
    int i, j; 

    FILE * fout = fopen(filename.c_str(), "w");
    if (!fout) {
      printf("Cannot open file %s to save!\n", filename.c_str());
      return 1;
    }

    // wirte docs with topic assignments for words
    for (i = 0; i < ptrndata->M; i++) {    
      for (j = 0; j < ptrndata->docs[i]->length; j++) {
	fprintf(fout, "w%d:a%d:t%d ", ptrndata->docs[i]->words[j], x[i][j], z[i][j]);
      }
      fprintf(fout, "\n");
    }

    fclose(fout);
    return 0;
  }

/*
  int model::save_model_tassign_for_author(string filename) {
    int i, j;
    
    FILE * fout = fopen(filename.c_str(), "w");
    if (!fout) {
      printf("Cannot open file %s to save!\n", filename.c_str());
      return 1;
    }

    // wirte docs with topic assignments for author
    for (i = 0; i < ptrndata->M; i++) {   
      for (j = 0; j < ptrndata->docs[i]->length; j++) { 
	fprintf(fout, "%d:%d ", ptrndata->docs[i]->words[j], x[i][j]);
      }
      fprintf(fout, "\n");
    }
    fclose(fout);
    return 0;
  }
*/
  //author topic model
  int model::save_model_theta(string filename) {
    FILE * fout = fopen(filename.c_str(), "w");
    if (!fout) {
      printf("Cannot open file %s to save!\n", filename.c_str());
      return 1;
    }

    for (int i = 0; i < A; i++) {
      for (int j = 0; j < K; j++) {
	fprintf(fout, "%f ", theta[i][j]);
      }
      fprintf(fout, "\n");
    }
   
    fclose(fout);
    return 0;
  }


  int model::save_model_phi(string filename) {
    FILE * fout = fopen(filename.c_str(), "w");
    if (!fout) {
      printf("Cannot open file %s to save!\n", filename.c_str());
      return 1;
    }
    
    for (int i = 0; i < K; i++) {
      for (int j = 0; j < V; j++) {
	fprintf(fout, "%f ", phi[i][j]);
      }
      fprintf(fout, "\n");
    }

    fclose(fout);    
    return 0;
  }

int model::save_model_others(string filename) {
  FILE * fout = fopen(filename.c_str(), "w");
  if (!fout) {
    printf("Cannot open file %s to save!\n", filename.c_str());
    return 1;
  }

  fprintf(fout, "alpha=%f\n", alpha);
  fprintf(fout, "beta=%f\n", beta);
  fprintf(fout, "ntopics=%d\n", K);
  fprintf(fout, "ndocs=%d\n", M);
  fprintf(fout, "nwords=%d\n", V);
  fprintf(fout, "nauthords=%d\n", A);
  fprintf(fout, "liter=%d\n", liter);
    
  fclose(fout);    
    
  return 0;
}

  int model::save_model_twords(string filename) {
    FILE * fout = fopen(filename.c_str(), "w");

    if (!fout) {
      printf("Cannot open file %s to save!\n", filename.c_str());
      return 1;
    }
    
    if (twords > V) {
      twords = V;
    }

    mapid2word::iterator it;
    for (int k = 0; k < K; k++) {
      vector<pair<int, double> > words_probs; //words_probs[word_prob,word_prob,,,,,]
      pair<int, double> word_prob;
      for (int w = 0; w < V; w++) {
	word_prob.first = w; //insert word id
	word_prob.second = phi[k][w]; //insert word_topic prob
	words_probs.push_back(word_prob); //insert word_probs
      }

      // quick sort to sort word-topic probability
      utils::quicksort(words_probs, 0, words_probs.size() - 1);
	
      fprintf(fout, "Topic %dth:\n", k);
      for (int i = 0; i < twords; i++) {
	it = id2word.find(words_probs[i].first);
	if (it != id2word.end()) {
	  fprintf(fout, "\t%s   %f\n", (it->second).c_str(), words_probs[i].second);
	}
      }
    }
    
    fclose(fout);    
    
    return 0;    
  }

  int model::save_model_tauthors(string filename) {
    FILE * fout = fopen(filename.c_str(), "w");
    if (!fout) {
      printf("Cannot open file %s to save!\n", filename.c_str());
      return 1;
    }

    if (tauthors > A) {
      tauthors = A;
    }
    mapid2author::iterator it;

    for (int k = 0; k < K; k++) {
      vector<pair<int, double> > authors_probs; //words_probs[word_prob,word_prob,,,,,]
      pair<int, double> author_prob;
      for (int a = 0; a < A; a++) {
	author_prob.first = a; //insert author id
	author_prob.second = theta[a][k]; //insert author_topic prob
	authors_probs.push_back(author_prob); //insert author_probs
      }
    
      // quick sort to sort word-topic probability
      utils::quicksort(authors_probs, 0, authors_probs.size() - 1);
           
      fprintf(fout, "Topic %dth:\n", k);
      for (int i = 0; i < tauthors; i++) {
	it = id2author.find(authors_probs[i].first);
	if (it != id2author.end()) {
	  fprintf(fout, "\t%s   %f\n", (it->second).c_str(), authors_probs[i].second);
	  printf("\t%s   %f\n", (it->second).c_str(), authors_probs[i].second);
	}
      }
    }    
    fclose(fout);    
    return 0;    
  }

  int model::init_est() {
    int i, m, n, w, k, a;
    // + read training data
    ptrndata = new dataset;
    if (ptrndata->read_trndata(dir + dfile, dir + wordmapfile, dir + authormapfile)) {
      printf("Fail to read training data!\n");
      return 1;
    }

    // + allocate memory and assign values for variables
    M = ptrndata->M;
    V = ptrndata->V;
    A = ptrndata->A;

    //pサンプリングに用いる確率分布
    p = new double*[A];
    for (i = 0; i < A; i++){
      p[i] = new double[K];
      for (k = 0; k < K; k++){
	p[i][k] = 0;
      }
    }

    // K: from command line or default value
    // alpha, beta: from command line or default values
    // niters, savestep: from command line or default values
    
    //各単語に割り振られたトピックの総和
    nw = new int*[V];
    for (w = 0; w < V; w++) {
      nw[w] = new int[K];
      for (k = 0; k < K; k++) {
	nw[w][k] = 0;
      }
    }

    //著者aに割り振られたトピックkの総和
    na = new int*[A];
    for (a = 0; a < A; a++) {
      na[a] = new int[K];
      for (k = 0; k < K; k++) {
	na[a][k] = 0;
      }
    }

    //トピックKの総和
    nwsum = new int[K];
    for (k = 0; k < K; k++) {
      nwsum[k] = 0;
    }

    //著者Aに割り振られたトピックの総和
    nasum = new int[A];
    for (a = 0; a < A; a++) {
      nasum[a] = 0;
    }

    // initialize for random number generation
    srandom(time(0));
    z = new int*[M];
    x = new int*[M];
 
    for (m = 0; m < ptrndata->M; m++) {
      int N = ptrndata->docs[m]->length;
      int doc_author_num = ptrndata->docs[m]->author_num;//文章mの著者数
      int document_author[doc_author_num];//文章mの著者数を格納する為の配列

      /*
      for(i=0; i< doc_author_num; i++){//文章mの著者を決定する
	document_author[i] = (int)(((double)random() / RAND_MAX) * A);
	}
      */

      for(i=0; i< doc_author_num; i++){//文章mの著者を決定する
	document_author[i] = ptrndata->docs[m]->authors[i];
      }
      
      z[m] = new int[N];
      x[m] = new int[N];

      // initialize for z & x
      for (n = 0; n < N; n++) {
	int topic = (int)(((double)random() / RAND_MAX) * K);
	//配列に格納された著者から文章の著者を選択する
	int selector= (int)(((double)random() / RAND_MAX) * doc_author_num);
	int author= document_author[selector];

	z[m][n] = topic;
	x[m][n] = author;
	
	// number of instances of word i assigned to topic j
	nw[ptrndata->docs[m]->words[n]][topic] += 1;

	// number of instances of author i assigned to topic j
	na[author][topic] += 1;

	// total number of words assigned to topic j
	nwsum[topic] += 1;
	nasum[author]+= 1;
      } 
    }

    theta = new double*[A];
    for (a = 0; a < A; a++) {
      theta[a] = new double[K];
    }

    phi = new double*[K];
    for (k = 0; k < K; k++) {
      phi[k] = new double[V];
    }    
    return 0;
  }

  //LDAの本体
  void model::estimate() {

    double max = -100000000;
    double new_logP = 0.0;
    int **max_x;
    int **max_z;

    if (twords > 0) {
      // print out top words per topic
      dataset::read_wordmap(dir + wordmapfile, &id2word);
    }
    if (tauthors > 0){
    // print out author per topic
    dataset::read_authormap(dir + authormapfile, &id2author);
    }
    //イテレーションの回数を表示
    printf("Sampling %d iterations!\n", niters);
    
    //一つ前のイテレーションの回数の続きから始める
    int last_iter = liter;
    for (liter = last_iter + 1; liter <= niters + last_iter; liter++) {
 
      printf("Iteration %d ...\n", liter);
      //m文章のn番目のトピックz_iを取り除いてサンプリングしたトピックをm文章のn番目のトピックとする
      // for all z_i
      for (int m = 0; m < M; m++) {
	for (int n = 0; n < ptrndata->docs[m]->length; n++) {
	  sampling(m, n, &z[m][n], &x[m][n]);
	}
      }
      //debug point 2
      new_logP = logP_zaw();
      max = save_max_logP(new_logP,  max, max_x, max_z);
    
      if (savestep > 0) {
	if (liter % savestep == 0) {//指定したステップ数まで終わったらモデルを一度保存する
	  // saving the model
	  printf("Saving the model at iteration %d ...\n", liter);
	  compute_theta();
	  compute_phi();
	  save_model(utils::generate_model_name(liter));
	}
      }
    }
    mapid2author::iterator it;
    //各文章のtopic確認
    FILE * fout = fopen("topic_show.txt", "w");
    for(int i=0; i < M; i++){
      for(int j=0; j < ptrndata->docs[i]->length; j++){
	it = id2word.find(ptrndata->docs[i]->words[j]);
	fprintf(fout, "[%s %d] ", (it->second).c_str(), z[i][j]);
	x[i][j] = max_x[i][j];
	z[i][j] = max_z[i][j];
      }
      fprintf(fout,"\n");
      fprintf(fout,"document number = %d \n",i);
    }

    printf("Gibbs sampling completed!\n");
    printf("Saving the final model!\n");
    compute_theta();
    compute_phi();
    liter--;
    save_model(utils::generate_model_name(-1));
  }

  void model::sampling(int m, int n, int *top, int *auth) {//m文書のn番目をサンプリング
    // remove z_i from the count variables
    int topic  = z[m][n];//サンプリングする箇所のトピックと著者
    int author = x[m][n];
    int a = 0;
    int w = ptrndata->docs[m]->words[n];
    int document_m_author = ptrndata->docs[m]->author_num;//ここでは文章mに登場する著者数をAとする
    int document_author[document_m_author];
    int C;

  //int a = ptrndata->docs[m]->authors[n];//check文章に割り当てられた著者
    nw[w][topic] -= 1;
    na[author][topic] -= 1;
    nwsum[topic] -= 1;
    nasum[author] -= 1;

    double Vbeta = V * beta;
    double Kalpha = K * alpha;
    
    for(int i=0; i< document_m_author; i++){ //配列に著者を入れる
      document_author[i] = ptrndata->docs[m]->authors[i];
    }

    // do multinomial sampling via cumulative method
    for(int i=0; i < document_m_author; i++){ 
      a = document_author[i];
      //debug point 1
      for (int k = 0; k < K; k++) {
	p[a][k] = (nw[w][k] + beta) / (nwsum[k] + V*beta) *
	  (na[a][k] + alpha) / (nasum[a] + K*alpha);
      }
    }

    // cumulate multinomial parameters
    // a = 0
    for(int k = 1; k < K; k++){
      p[document_author[0]][k] += p[document_author[0]][k-1];
    }
    if(document_m_author > 1 && (document_author[0] != document_author[1]) ){
      // a = 1~document_m_author
      for(int i = 1; i < document_m_author; i++){
	a = document_author[i];
	for(int k = 0; k < K; k++){
	  if(k == 0) p[a][k] += p[document_author[i -1]][K-1];
	  else p[a][k] += p[a][k-1];
	}
      }
    }
  
    //printf("p[%d][%d]=%f\t", document_m_author-1, K-1, p[document_author[document_m_author-1]][K-1]);
    //printf("\n");
    // scaled sample because of unnormalized p[]
    double u = ((double)random() / RAND_MAX) * p[document_author[document_m_author - 1]][K - 1];
    //printf("p[document_author[document_m_author - 1]][K-1] = %f\n",p[document_author[document_m_author - 1]][K-1]);
    for (author = 0; author < document_m_author; author++){   
      for (topic = 0; topic < K; topic++) {
	//printf("p[[%d][%d]] = %f u=%f\n", author, topic, p[document_author[author]][topic], u);
	if (u == 1) u = ((double)(random()-1) / RAND_MAX) * p[document_author[document_m_author - 1]][K - 1];
	if (p[document_author[author]][topic] > u) {
	 
	  goto OUT;
	}
      }
    }
  OUT:
    //printf("author = %d topic = %d \n\n\n", author, topic);
    author = document_author[author];

    /*あり得ない著者ID若しくはトピックIDが出現した時は最大の著者とトピックを返す.
    if( author > A && topic > K){
      author = A-1;
      topic  = K-1;
      }*/

    //author = ptrndata->docs[m]->authors[author];
    //printf("author = %d topic = %d \n\n\n", author, topic);    
    // add newly estimated z_i to count variables
    nw[w][topic] += 1;
    na[author][topic] += 1;
    nasum[author] += 1;
    nwsum[topic] += 1;
   
    *top = topic;
    *auth = author;
  }

  void model::compute_theta() {
    for (int a = 0; a < A; a++) {
      for (int k = 0; k < K; k++) {
	theta[a][k] = (na[a][k] + alpha) / (nasum[a] + K * alpha);
      }
    }
  }

  void model::compute_phi() {
    for (int k = 0; k < K; k++) {
      for (int w = 0; w < V; w++) {
	phi[k][w] = (nw[w][k] + beta) / (nwsum[k] + V * beta);
      }
    }
  }

/* computes log-gamma function */
// Γ(1.0)=Γ(2.0)=1.0 を考慮
//  （∵loggamma() は loggamma(1) や loggamma(2) のとき，本来は 0 であるが，
//      実際の計算では非常に絶対値の小さな負の値になってしまうため）
double model::LogGamma(double x)
{
  if(x == 1 || x==2) return(0);

  double gv, gw, lg;
  gv = 1;
  while(x < 8){ gv *= x; x++;}
  gw = 1 / (x * x);
  lg = (((((((( (-3617.0 / 510.0) / (16 * 15))  * gw + ( ( 7.0 / 6.0) / (14 * 13))) * gw
	          +  ( (-691.0 / 2730.0) / (12 * 11))) * gw + ( ( 5.0 / 66.0) / (10 *  9))) * gw
	          +  ( (-1.0 / 30.0)  / ( 8 *  7))) * gw + ( ( 1.0 / 42.0)  / ( 6 *  5))) * gw
	          +  ( (-1.0 / 30.0)  / ( 4 *  3))) * gw + ( ( 1.0 / 6.0)  / ( 2 *  1))) / x
                  + 0.5 * 1.83787706640934548 - log(gv) - x + (x - 0.5) * log(x);
	return(lg);
}

// log P(vec{z}, vec{x}, vec{w} | α, β, γ, δ) の計算
//
//  P(z,x|w) ∝ P(z,x,w)
//
//         = η1^M * Π_m η0^{Nm - 1} *     (Nm  : doc m の長さ)
//
//                    Γ(Kα)     Π_k Γ(NP_DZ(m,k)+α)
//           ・ Π_m --------- * ---------------------- 
//                   Γ(α)^K     Γ(SumNP_DZ(m)+Kα)
// 
//                   Γ(Vβ)     Π_w Γ(NPH_ZW(k,w)+β)
//           ・ Π_k --------- * ---------------------- 
//                   Γ(β)^V     Γ(SumNPH_ZW(k)+Vβ)
// 
//                         Γ(Vγ)    Π_w2 Γ(N_WpZW(w,k,w2)+γ)
//           ・ Π_w Π_k --------- * --------------------------- 
//                        Γ(γ)^V      Γ(SumN_WpZW(w,k)+Vγ)
// 
//                          Γ(δ0+δ1)     Γ(N_WZX(w,k,0)+δ0)*Γ(N_WZX(w,k,1)+δ1)
//           ・ Π_w Π_k --------------- * ----------------------------------------- 
//                        Γ(δ0)*Γ(δ1)         Γ(SumN_WZX(w,k)+δ0+δ1)
//
//
//  であるから，
//
//  log P(z,x|w)
//      = M * [ logΓ(Kα) - K*logΓ(α) ]
//            + ∑_m [ {∑_k logΓ(NP_DZ[m][k] + α)} - logΓ(SumNP_DZ[m] + Kα) ]
//         + K * [ logΓ(Vβ) - V*logΓ(β) ]
//            + ∑_k [ {∑_w logΓ(NPH_ZW[k][w] + β)} - logΓ(SumNPH_ZW[k] + Vβ) ]
//         + VK * [ logΓ(Vγ) - V*logΓ(γ)  ]
//            + ∑_w ∑_k [ {∑_w2 logΓ(N_WpZW[w][k][w2] + γ)} - logΓ(SumN_WpZW[w][k] + Vγ) ]
//         + VK * [ logΓ(δ0+δ1) - logΓ(δ0) - logΓ(δ1) ] 
//            + ∑_w ∑_k [ logΓ(N_WZX[w][k][0] + δ0) + logΓ(N_WZX[w][k][1] + δ1) - logΓ(SumN_WZX[w][k] + δ0 + δ1) ]
//         + C
//  実際には大小の比較だけに利用するため，C は計算しない

//logPの計算
double model::logP_zaw(){
  double logP = 0;
  double LogGamma_alpha = LogGamma(alpha);
  double Log_Const = 0;

  Log_Const += K*(LogGamma(V*beta) - V*LogGamma(beta));
  Log_Const += A*(LogGamma(K*alpha) - K*LogGamma(alpha));
  for(int m=0; m<M; m++){
    Log_Const += 1;
    Log_Const -= ptrndata->docs[m]->length * log(ptrndata->docs[m]->author_num);
  }

  for(int a=0 ; a<A ; a++){
    for(int k=0 ; k<K ; k++){
      if(na[a][k]==0){
	logP += LogGamma_alpha;
      }
      else{
	logP += LogGamma(na[a][k]+alpha);
      }
    }
    //    printf("logP = %f\n", logP);    
    logP -= LogGamma(nasum[a]+K*alpha);
  }

  double LogGamma_beta = LogGamma(beta);

  for(int k=0 ; k<K ; k++){
    for(int w=0 ; w<V ; w++){
      if(nw[w][k]==0)
	logP += LogGamma_beta;
      else
	logP += LogGamma(nw[w][k]+beta);
    }
    logP -= LogGamma(nwsum[k]+V*beta);
  }

  return logP + Log_Const;
}


int model::save_max_logP(double logP, double ex_max, int **max_x, int **max_z){
  
  double max  = 0.0;
  int document_num = ptrndata->M;
  
  max_x = new int*[document_num];
  max_z = new int*[document_num];

  if(logP > ex_max){
    max = logP;
    printf("***logP %f\n",max);
    for(int i=0; i< document_num; i++){
      int N = ptrndata->docs[i]->length;
      max_x[i] = new int[N];
      max_z[i] = new int[N];
      for(int j=0; j<N; j++){
	max_x[i][j] = x[i][j];
	max_z[i][j] = z[i][j];
      }
    }
  }
  else max = ex_max;

  return max;
}
