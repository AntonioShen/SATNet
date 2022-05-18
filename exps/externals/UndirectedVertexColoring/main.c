#include <stdio.h>
#include <stdlib.h>
#define INFINITY 255   //表示图中两个顶点无关联

/**
 * Reference: https://zhuanlan.zhihu.com/p/111138152
 */
typedef struct{
    int vertexNum; //vertex numbers
    int arcNum;    //edge numbers
    int *arc;      //points to the array of graph's relations
    int *color;    //points to the vertex coloring in the graph
}Graph;
void creatGraph(Graph *g);
int ColorGraph(Graph *g);
int Conflict(Graph *g,int k);
void dispVertexColor(Graph *g);
int main()
{
	Graph g;   //declare a graph g
	creatGraph(&g);//create a graph based on the vertex number, edge number and relations between vertices
	int m=ColorGraph(&g);
	printf("%d",m);
	dispVertexColor(&g);
	return 0;
}
/**
 * 以下creatGraph()函数
 * 定义图的定点数、边数、
 * 图的邻接矩阵、顶点着色记录数组
 **/
void creatGraph(Graph *g)
{
    int x,y,k,w;
    scanf("%d %d",&g->vertexNum,&g->arcNum);
    g->arc=(int*)malloc(sizeof(int)*g->vertexNum*g->vertexNum);//创建邻接矩阵
    g->color=(int*)malloc(sizeof(int)*g->vertexNum);//创建顶点着色记录数组
    if(g->vertexNum==1&&g->arcNum==0)
    {
        g->arc[0]=0;
        g->color[0]=1;
        return;
    }
    for(x = 0;x < g->vertexNum;x++){//初始化邻接矩阵中的元素为INFINITY，即顶点之间不连通
        for(y = 0;y < g->vertexNum;y++)
            g->arc[g->vertexNum * x + y]=INFINITY;
        g->color[x]=0;//初始化着色记录数组元素为0，即无色。
    }
    for (k = 0; k < g->arcNum; k++){ //关联矩阵为一个对称矩阵
        scanf("%d %d %d",&x,&y,&w);
        g->arc[g->vertexNum*x+y]=w; //将关联边长度存入图的矩阵x行y列
        g->arc[g->vertexNum*y+x]=w; //将关联边长度存入图的矩阵y行x列
    }
}
void dispVertexColor(Graph *g)       //输出图g中各顶点的着色情况函数
{
    printf("[");
    for(int i = 0;i < g->vertexNum - 1; i++)
        printf("%d,",g->color[i]);
    printf("%d]\n",g->color[g->vertexNum - 1]);
}

int ColorGraph(Graph *g)   //给图的顶点着色，函数返回值为最小着色数的值。
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
int Conflict(Graph *g,int k)   //检测顶点k与其邻接点着色是否冲突检，返回1为不冲突，返回0为冲突，此函数由ColorGraph()函数调用
{
	for(int i=0;i<k;i++)
 	{
		if(g->arc[g->vertexNum*k+i]!=255&&g->color[i]==g->color[k])
			return 0;
	}
	return 1;
}
