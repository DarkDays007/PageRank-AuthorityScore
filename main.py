import networkx as nx
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
import numpy as np



def load_graph(file_path):
  
    with open(file_path, 'r') as file:  

         lines = file.readlines()   

    edges = set()

    nodes = {}
    source = None

    for line in lines:


        line = line.strip()
        if line.startswith('edge'): 
             
            continue
        if 'source' in line or 'target' in line:

            key, value = line.split(' ')

            value = value.strip('"')   
            if key == 'source':   
                source = value

            elif key == 'target':
                target = value
                edges.add((source, target))

        if 'id' in line or 'value' in line:

            key, value = line.split(' ')      
            value = value.strip('"')  


            if key == 'id':

                current_node = value

                nodes[current_node] = {}


            elif key == 'value':

                nodes[current_node]['value'] = int(value)  

    
    graph = nx.DiGraph()
    graph.add_nodes_from([(node, attrs) for node, attrs in nodes.items()])



    graph.add_edges_from(edges)   
    return graph






def basic_analysis(graph):
    print("Number of nodes:", graph.number_of_nodes())      
    print("Number of edges:", graph.number_of_edges())

    categories = nx.get_node_attributes(graph, 'value')

    liberal = sum(1 for v  in categories.values() if v == 0) 

    conservative = sum(1 for v in categories.values() if v == 1)

    print("Liberal  blogs:", liberal)
    print("Conservative blogs:", conservative)

    in_degrees = [graph.in_degree(n) for n in graph.nodes()]

    out_degrees = [graph.out_degree(n) for n in graph.nodes()]

    print("Min  in-degree:", min(in_degrees))
    print("Max in-degree:", max(in_degrees))
    print("Average in-degree:", np.mean(in_degrees))
    print("Min  out-degree:", min(out_degrees))
    print("Max out-degree:", max(out_degrees))
    print("Average out-degree:", np.mean(out_degrees))



def weighted_pagerank_category(graph, alpha=0.85):

    categories = nx.get_node_attributes(graph, 'value')

    for u, v in graph.edges():

        graph[u][v]['weight'] = 2 if categories[u] == categories[v] else 1 
    return nx.pagerank(graph, alpha=alpha, weight='weight')

def weighted_pagerank_reciprocal(graph, alpha=0.85): 
    for u, v in graph.edges():

        graph[u][v]['weight'] = 2 if graph.has_edge(v, u) else 1
    return nx.pagerank(graph, alpha=alpha, weight='weight')         


def hits_with_degree_normalization(graph, max_iter=100, tol=1e-8):
      
    hubs, authorities = nx.hits(graph, max_iter=max_iter, tol=tol, normalized=True) 
    norm_hubs = {k: v / max(graph.out_degree(k, 1), 1) for k, v in hubs.items()}

    norm_auths = {k: v / max(graph.in_degree(k, 1), 1) for k, v in authorities.items()}    

    return norm_hubs, norm_auths




def aggregate_scores(scores, categories):

    aggregate = {0: 0, 1: 0}     
    for node, score in  scores.items():

        aggregate[categories[node]] += score
    return aggregate



def calculate_spearman(scores1, scores2):             
    keys = list(scores1.keys())

    ranks1 = [scores1[k] for k in keys]
    ranks2 = [scores2[k] for k in keys]
     
    return spearmanr(ranks1, ranks2).correlation
def calculate_overlap(scores1, scores2, top_k=10):

    top1 = set(sorted(scores1, key=scores1.get, reverse=True)[:top_k])

    top2 = set(sorted(scores2, key=scores2.get, reverse=True)[:top_k])
    return len(top1.intersection(top2)) / top_k





def plot_scores_distribution(scores, title):


    plt.hist(scores.values(), bins=20, alpha=0.75)
    plt.title(title)
    plt.xlabel("Scores")
    plt.ylabel("Frequency")
    plt.show()


def run_experiments(file_path):

    graph = load_graph(file_path)
    basic_analysis(graph)

    print("\nRunning PageRank with different betas...")

    pr_beta_1 = nx.pagerank(graph, alpha=0.6)                                         
    pr_beta_2 = nx.pagerank(graph, alpha=0.8)      
    pr_beta_3 = nx.pagerank(graph, alpha=0.9)            

    print("Spearman Correlations between different betas:")              
    print("0.6 vs 0.8:", calculate_spearman(pr_beta_1, pr_beta_2))        
    print("0.8 vs 0.9:", calculate_spearman(pr_beta_2, pr_beta_3))                                         

    print("\nRunning PageRank variations...")
    pr_category = weighted_pagerank_category(graph)
    pr_reciprocal = weighted_pagerank_reciprocal(graph)

    print("\nRunning HITS variation...")
    hubs, authorities = hits_with_degree_normalization(graph)

    print("\nAggregating scores by category...")
    categories = nx.get_node_attributes(graph, 'value')


    pr_category_agg = aggregate_scores(pr_category, categories)    


    pr_reciprocal_agg = aggregate_scores(pr_reciprocal, categories)


    authority_agg = aggregate_scores(authorities, categories)            



    print("\nCategory Aggregates:")       
    print("PageRank (Category):", pr_category_agg)      
    print("PageRank (Reciprocal):", pr_reciprocal_agg)
    print("Authorities:", authority_agg)        

    print("\nCalculating Spearman correlations and overlaps...")

    spearman_pr = calculate_spearman(pr_category, pr_reciprocal)  
    overlap_pr = calculate_overlap(pr_category, pr_reciprocal) 


    print("Spearman Correlation (PR Category vs PR Reciprocal):", spearman_pr)
    print("Top-10 Overlap (PR Category vs PR Reciprocal):", overlap_pr)        

    spearman_hits = calculate_spearman(hubs, authorities)    
    overlap_hits = calculate_overlap(hubs, authorities) 

    print("Spearman Correlation (Hubs vs Authorities):", spearman_hits)
    print("Top-10 Overlap (Hubs vs Authorities):", overlap_hits)         

    print("\nPlotting distributions...")

    plot_scores_distribution(pr_category, "PageRank Category Scores")       
    plot_scores_distribution(pr_reciprocal, "PageRank Reciprocal Scores")       
    plot_scores_distribution(authorities, "Authority Scores")         


#workflow
run_experiments('polblogs.gml')
