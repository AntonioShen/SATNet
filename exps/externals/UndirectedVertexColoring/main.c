#include <stdio.h>
#include <stdlib.h>
#define INFINITY 255   //means two vertices in the graph has no relation

/**
 * Reference: https://zhuanlan.zhihu.com/p/111138152
 */
typedef struct{
    int vertexNum; //vertex numbers
    int arcNum;    //edge numbers
    int *arc;      //points to the array of graph's relations
    int *color;    //points to the vertex coloring in the graph
}Graph;
void createGraph(Graph *g);
int ColorGraph(Graph *g);
int Conflict(Graph *g,int k);
void dispVertexColor(Graph *g);
const int size = 10; //defines the number of vertices in the graph
int main()
{
//    FILE *adj_list;
//    adj_list = fopen("../../../myexps/adj_list.sst", "r");
//    char * line = NULL;
//    size_t len = 0;
//    while (getline(&line, &len, adj_list)) {
//        printf("%s", line);
//    }
//    fclose(adj_list);
	Graph g; //declare a graph g
	createGraph(&g); //create a graph based on the vertex number, edge number and relations between vertices
	int m=ColorGraph(&g);
	printf("%d",m);
	dispVertexColor(&g);
	return 0;
}
/**
 * The following creatGraph() function
 * defines the graph's vertex number and edge number
 * The graph's adjacency matrix and the coloring array
 **/
void createGraph(Graph *g)
{
    int x,y,k,w;
    FILE *adj_list;
    adj_list = fopen("../../../myexps/adj_list.sst", "r");
    char * line = NULL;
    size_t len = 0;
    getline(&line, &len, adj_list);
    sscanf(line, "%d %d",&g->vertexNum,&g->arcNum);
    g->arc=(int*)malloc(sizeof(int)*g->vertexNum*g->vertexNum); //create adjacency matrix
    g->color=(int*)malloc(sizeof(int)*g->vertexNum); //create coloring array
    if(g->vertexNum==1&&g->arcNum==0)
    {
        g->arc[0]=0;
        g->color[0]=1;
        return;
    }
    for(x = 0;x < g->vertexNum;x++){ //the element in the initial adjacency matrix is INFINITY, that is, the vertices are not connected
        for(y = 0;y < g->vertexNum;y++)
            g->arc[g->vertexNum * x + y]=INFINITY;
        g->color[x]=0; //initialize the shaded record array element to 0, i.e., colorless
    }
    for (k = 0; k < g->arcNum; k++){ //the correlation matrix is a symmetric matrix
        getline(&line, &len, adj_list);
        sscanf(line, "%d %d %d",&x,&y,&w);
        g->arc[g->vertexNum*x+y]=w; //store the associated edge length in the matrix x row y column of the graph
        g->arc[g->vertexNum*y+x]=w; //store the associated edge length in the matrix y row x column of the graph
    }
    fclose(adj_list);
}
void dispVertexColor(Graph *g) //output the coloring function of each vertex in graph g
{
    printf("[");
    for(int i = 0;i < g->vertexNum - 1; i++)
        printf("%d,",g->color[i]);
    printf("%d]\n",g->color[g->vertexNum - 1]);
}

int ColorGraph(Graph *g) //to color the vertex of a graph, the function returns the minimum chromatic number
{
  int k=0,m=1;
  if(g->vertexNum==1 && g->arcNum==0)
  {
	  g->color[0]=1;
	  return m;
  }
  while(1)
  {
    while(k>=0)
    {
        g->color[k]=g->color[k]+1;
	    while(g->color[k]<=m)
	    {
            if(Conflict(g,k)) break;
		    else g->color[k]=g->color[k]+1;
	    }
	    if(g->color[k]<=m && k==g->vertexNum-1)
	        return m;
	    if(g->color[k]<=m && k<g->vertexNum-1)
	        k=k+1;
	    else
		    g->color[k--]=0;
    }
  k=0;
  m++;
  }
}
int Conflict(Graph *g,int k) //this function is called by ColorGraph() to check whether vertex K and its adjacent point shaders clash. Returns 1 for no conflict and 0 for conflict
{
	for(int i=0;i<k;i++)
 	{
		if(g->arc[g->vertexNum*k+i]!=255&&g->color[i]==g->color[k])
			return 0;
	}
	return 1;
}
