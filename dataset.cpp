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

#include <stdio.h>
#include <stdlib.h>
#include "constants.h"
#include "strtokenizer.h"
#include "dataset.h"

using namespace std;

int dataset::show_model_id(string filename, document * pdoc){
  FILE * fout = fopen(filename.c_str(), "w");
  if(!fout){
    printf("Cannot open file %s to save!\n", filename.c_str());
    return 1;
  }
  for(int i=0; i < pdoc->length; i++){
    fprintf(fout, "%d ",pdoc->words[i]);
    }
  fprintf(fout, "\n");
  for(int i=0; i < pdoc->author_num; i++){
    fprintf(fout, "%d ",pdoc->authors[i]);
  }

  fclose(fout);
  return 0;
}

int dataset::write_wordmap(string wordmapfile, mapword2id * pword2id) {
    FILE * fout = fopen(wordmapfile.c_str(), "w");
    if (!fout) {
	printf("Cannot open file %s to write!\n", wordmapfile.c_str());
	return 1;
    }    
    
    mapword2id::iterator it;
    fprintf(fout, "%d\n", pword2id->size());
    for (it = pword2id->begin(); it != pword2id->end(); it++) {
	fprintf(fout, "%s %d\n", (it->first).c_str(), it->second);
    }
    
    fclose(fout);
    
    return 0;
}

int dataset::read_wordmap(string wordmapfile, mapword2id * pword2id) {
    pword2id->clear();
    
    FILE * fin = fopen(wordmapfile.c_str(), "r");
    if (!fin) {
	printf("Cannot open file %s to read!\n", wordmapfile.c_str());
	return 1;
    }    
    
    char buff[BUFF_SIZE_SHORT];
    string line;
    
    fgets(buff, BUFF_SIZE_SHORT - 1, fin);
    int nwords = atoi(buff);
    
    for (int i = 0; i < nwords; i++) {
	fgets(buff, BUFF_SIZE_SHORT - 1, fin);
	line = buff;
	
	strtokenizer strtok(line, " \t\r\n");
	if (strtok.count_tokens() != 2) {
	    continue;
	}
	
	pword2id->insert(pair<string, int>(strtok.token(0), atoi(strtok.token(1).c_str())));
    }
    
    fclose(fin);
    
    return 0;
}

int dataset::read_wordmap(string wordmapfile, mapid2word * pid2word) {
    pid2word->clear();
    
    FILE * fin = fopen(wordmapfile.c_str(), "r");
    if (!fin) {
	printf("Cannot open file %s to read!\n", wordmapfile.c_str());
	return 1;
    }    
    
    char buff[BUFF_SIZE_SHORT];
    string line;
   
    fgets(buff, BUFF_SIZE_SHORT - 1, fin);
    int nwords = atoi(buff);
    
    for (int i = 0; i < nwords; i++) {
	fgets(buff, BUFF_SIZE_SHORT - 1, fin);
	line = buff;
	
	strtokenizer strtok(line, " \t\r\n");
	if (strtok.count_tokens() != 2) {
	    continue;
	}
	
	pid2word->insert(pair<int, string>(atoi(strtok.token(1).c_str()), strtok.token(0)));
    }
    
    fclose(fin);
    return 0;
}

int dataset::read_authormap(string wordmapfile, mapid2author * pid2word) {
    pid2word->clear();
    
    FILE * fin = fopen(wordmapfile.c_str(), "r");
    if (!fin) {
	printf("Cannot open file %s to read!\n", wordmapfile.c_str());
	return 1;
    }    
    
    char buff[BUFF_SIZE_SHORT];
    string line;
   
    fgets(buff, BUFF_SIZE_SHORT - 1, fin);
    int nwords = atoi(buff);
    
    for (int i = 0; i < nwords; i++) {
	fgets(buff, BUFF_SIZE_SHORT - 1, fin);
	line = buff;
	
	//	strtokenizer strtok(line, "+\t\r\n"); 英語テキスト用　生徒名の間にスペースがあるため、人名をプラスで区切る
	strtokenizer strtok(line, " \t\r\n"); 
	if (strtok.count_tokens() != 2) {
	    continue;
	}
	
	pid2word->insert(pair<int, string>(atoi(strtok.token(1).c_str()), strtok.token(0)));
    }
    
    fclose(fin);
    return 0;
}

int dataset::write_authormap(string wordmapfile, mapauthor2id * pword2id) {
    FILE * fout = fopen(wordmapfile.c_str(), "w");
    if (!fout) {
	printf("Cannot open file %s to write!\n", wordmapfile.c_str());
	return 1;
    }    
    
    mapauthor2id::iterator it;
    fprintf(fout, "%d\n", pword2id->size());
    for (it = pword2id->begin(); it != pword2id->end(); it++) {
      //fprintf(fout, "%s+%d\n", (it->first).c_str(), it->second); 英語用
      fprintf(fout, "%s %d\n", (it->first).c_str(), it->second);
    }
    
    fclose(fout);
    
    return 0;
}



/*------------------------------------------------
引数に strint authormapを追加
著者トークンを格納するauthortok
著者の合計値を格納するauthor_num
mapauthor2id author2idを追加
----------------------------------------------------- */

int dataset::read_trndata(string dfile, string wordmapfile, string authormapfile) {
  mapword2id word2id;
  mapauthor2id author2id;

  FILE * fin = fopen(dfile.c_str(), "r");
  if (!fin) {
    printf("Cannot open file %s to read!\n", dfile.c_str());
    return 1;
  }   
     
  mapword2id::iterator it;    
  mapauthor2id::iterator ait;
  char buff[BUFF_SIZE_LONG];
  string line;
    
  // get the number of documents ?? File first line
  fgets(buff, BUFF_SIZE_LONG - 1, fin);// char*fgets(char *row,int len, FILE *fp) ファイルポインタからlenバイト読み込んでその先頭アドレスをrowに返す
  M = atoi(buff);//文字列をint型に変換する

  if (M <= 0) {
    printf("No document available!\n");
    return 1;
  }
    
  // allocate memory for corpus
  if (docs) {
    deallocate();
  } else {
    docs = new document*[M];
  }
  
  // set number of words & authors to zero
  V = 0;
  A = 0;
  for (int i = 0; i < M; i++) {

    fgets(buff, BUFF_SIZE_LONG - 1, fin);
    line = buff;
    strtokenizer strtok(line, "\t\r\n ");
    int length = strtok.count_tokens();
    
    if (length <= 0) {
      printf("Invalid (empty) document!\n");
      deallocate();
      M = V = 0;
      return 1;
    }
    
    fgets(buff, BUFF_SIZE_LONG - 1, fin);
    line = buff;
    //    strtokenizer authortok(line, "\t\r\n+");英語テキスト用　生徒名の間にスペースがあるため、人名をプラスで区切る
    strtokenizer authortok(line, "\t\r\n ");
    //    strtokenizer authortok(line, "+");
    int author_num = authortok.count_tokens();
    
    //著者が出現しない場合もあるので要考慮
    if (length <= 0) {
      printf("Invalid (empty) document!\n");
      deallocate();
      M = A = 0;
      return 1;
      }
    
    // allocate new document
    document * pdoc = new document(length, author_num);
 
    for (int j = 0; j < length; j++) {
      it = word2id.find(strtok.token(j));
      if (it == word2id.end()) {
	// word not found, i.e., new word
	pdoc->words[j] = word2id.size();
	word2id.insert(pair<string, int>(strtok.token(j), word2id.size()));
      } else {
	pdoc->words[j] = it->second;
      }
    }

    for (int k = 0; k < author_num; k++) {
      ait = author2id.find(authortok.token(k));
      if (ait == author2id.end()) {
	// author not found, i.e., new author
	pdoc->authors[k] = author2id.size();
	author2id.insert(pair<string, int>(authortok.token(k), author2id.size()));
      } else {
	pdoc->authors[k] = ait->second;
      }
    }

    show_model_id("show.txt", pdoc);

    // add new doc to the corpus
    add_doc(pdoc, i);
  }

  fclose(fin);
  // write word map to file
  if (write_wordmap(wordmapfile, &word2id)) {
    return 1;
  }

  if (write_authormap(authormapfile, &author2id)){
    return 1;
  }
 
  // update number of words
  V = word2id.size();
  A = author2id.size();
 
  return 0;
}

