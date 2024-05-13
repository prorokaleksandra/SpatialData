import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.neighbors import radius_neighbors_graph
import numpy as np
import standardise as se
from scipy.sparse.csgraph import connected_components
from matplotlib.ticker import PercentFormatter
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

st.set_option('deprecation.showPyplotGlobalUse', False)

#Creating radius neighbourhood graph
def create_graph(dataframe, radius, celltype=False):
    if celltype == False:
        df = dataframe
    else:
        df = dataframe[dataframe["celltype"] == celltype]
    coordinates_list = df[['nucleus.x', 'nucleus.y']].values
    info_list = df[['cell.ID', 'celltype']].values
    all_info = df[['nucleus.x', 'nucleus.y', 'cell.ID', 'celltype']].values
    graph = radius_neighbors_graph(coordinates_list, radius, mode='distance', metric='minkowski', p=2, metric_params=None, include_self=False)
    return graph, coordinates_list, info_list, all_info


#Streamlit function
def main():
    st.title('mIF spatial data analysis')
    filename = st.sidebar.selectbox('Select a file', os.listdir("./dane/if_data"))
    st.write('You selected `%s`' % filename)

    #Reading data and mapping the cell types
    df = pd.read_csv(os.path.join("./if_data", filename))

    df_mapping = pd.read_csv("./dane/IF1_phen_to_cell_mapping.csv")
    df['phenotype'] = df['phenotype'].apply(se.standardize_phenotype)
    merged_df = pd.merge(df, df_mapping, on='phenotype', how='left')
    min_x = merged_df["nucleus.x"].min()
    max_x = merged_df["nucleus.x"].max()
    min_y = merged_df["nucleus.y"].min()
    max_y = merged_df["nucleus.y"].max()

    #Scatterplot with all celltypes
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(x="nucleus.x", y="nucleus.y", data=merged_df, hue="celltype")
    ax.set_xlim(min_x*0.98, max_x*1.02)
    plt.title('Scatterplot of cell data')
    plt.xlabel("x_column")
    plt.ylabel("y_column")
    plt.legend(title="Cell type", loc='upper left')
    plt.grid(True)
    st.pyplot(fig)


    #Creating scatterplot with selectetypes
    fig, ax = plt.subplots()

    # Function creating scatterplot for certain celltype
    def generate_scatterplot(selected_cell_types):
        if not selected_cell_types:
            st.write("Select at least one cell type.")
            return
        ax.set_xlim(min_x*0.98, max_x*1.02)
        merged_data = merged_df[merged_df["celltype"].isin(selected_cell_types)]   
        sns.scatterplot(x="nucleus.x", y="nucleus.y", hue="celltype", data=merged_data, ax=ax)
        plt.title('Scatterplot of cell data')
        plt.xlabel("x_column")
        plt.ylabel("y_column")
        plt.legend(title="Cell type", loc='upper left')
        plt.grid(True)
        st.pyplot(fig)

    selected_cell_types = st.sidebar.multiselect("Select cell types", merged_df["celltype"].unique())

    generate_scatterplot(selected_cell_types)

    #Creating radius neighbourhood graphs for all cells and only bcells
    radius = st.sidebar.slider('Select radius:', min_value=1, max_value=50, value=30)
    graph_bcell, coordinates_bcell, info_bcell, info_with_coordinates_bcell = create_graph(merged_df, radius, "Bcell")
    graph_all, coordinates_all, info_all, info_with_coordinates= create_graph(merged_df, radius)

    #Finding connected components from bcells 
    n_components_bcell, labels_bcell = connected_components(csgraph=graph_bcell, directed=False, return_labels=True)

    #Counting how many cells are in components and filtering out those with less than 15
    counts_bcell = np.bincount(labels_bcell)
    threshold = 15
    indices = np.where(counts_bcell > threshold)[0]
    
    cells_dict ={} 
    for index in indices: 
        cells_dict[int(index)] = np.where(labels_bcell == index)[0]


    #Creating the dict where the id is the number of the component 
    #and the value is the list of the cells that belongs to this component
    component_dict = {}
    for component in cells_dict.keys():
        component_dict[component] = []
        index_list = []
        for cell in cells_dict[component]: 
            for index, cell_all in enumerate(info_all):
                if info_all[index][0] == info_bcell[cell][0]:
                    index_list.append(index)

        for target_index in index_list:
            #List of neighbours for index
            neighbors_indices = np.where(graph_all[target_index].toarray()[0] != 0)[0]
            for index in neighbors_indices:
                #Adding a tuple with the index and celltype and then deleting duplicates by creating the set
                component_dict[component].append(tuple(info_with_coordinates[index])) 
        #Creating the dict where the key is the index of component and the value is the list of types of cells        
        component_dict[component] = list(set(component_dict[component]))

    # Creating new df with info about component
    final_data = []
    for component, cell_list in component_dict.items():
        for cell_info in cell_list:
            final_data.append({
                'x.nucleus': cell_info[0],
                'y.nucleus': cell_info[1],
                'celltype': cell_info[3],
                'component': component
            })

    final_df = pd.DataFrame(final_data)


    # Scatterplot to differentiate cells from each component
    fig2, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(x='x.nucleus', y='y.nucleus', hue='component', data=final_df, palette='tab10')
    ax.set_xlim(min_x*0.98, max_x*1.02)
    ax.set_ylim(min_y*0.98, max_y*1.02)
    plt.title('Scatterplot of Cells by Component')
    plt.xlabel('x.nucleus')
    plt.ylabel('y.nucleus') 
    plt.legend(title='Component')
    plt.grid(True)
    st.pyplot(fig2)

    
    #Barplots with percentage of celltype in each component
    celltype_percents = {}
    for component in component_dict.keys(): 
        celltype_percents[component]={}
        celltypes = [celltype for x, y, index, celltype in component_dict[component]]
        total_cells = len(celltypes)
        unique_celltypes = list(set(celltypes))

        counts = []
        for celltype in unique_celltypes:
            counts.append(celltypes.count(celltype) / total_cells)
            celltype_percents[component][celltype] = celltypes.count(celltype) / total_cells

         # Obliczanie procentowego udziału
        fig2, ax2 = plt.subplots(figsize=(10, 6))  # Ustawienie rozmiaru figury na 10x6 cali

        plt.bar(unique_celltypes, counts, color='pink')  # Użycie funkcji plt.bar() do stworzenia wykresu słupkowego
        plt.title(f"Percentage of each celltype in component {component}")
        plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
        plt.xlabel('Celltype')
        plt.ylabel('Percentage')
        plt.xticks(rotation=30, ha='center')
        st.pyplot(fig2)
        

    # Creating list of unique celltypes and sorting it
    unique_cell_types = sorted(set(cell_type for component_data in celltype_percents.values() for cell_type in component_data.keys()))
    

    samples = []
    component_keys = []
    for component_key, component_data in celltype_percents.items():
        component_keys.append(component_key)
        sample = [component_data.get(cell_type, 0) for cell_type in unique_cell_types]
        samples.append(sample)


    # Conversion to a numpy array
    samples_array = np.array(samples)

    # Number of clusters
    n_clusters = 3

    # K-means clustering
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(samples_array)

    #PCA to reduce the number of dimensions
    pca = PCA(n_components=2)
    samples_pca = pca.fit_transform(samples_array)
    centers_pca = pca.transform(kmeans.cluster_centers_)

    #Components for each cluster
    components_in_clusters = {i: [] for i in range(n_clusters)}
    for i, label in enumerate(kmeans.labels_):
        components_in_clusters[label].append(component_keys[i])


    #Streamlit visualisation
    st.title('Wizualizacja środków klastrów za pomocą PCA')
    st.write('Wykres punktowy przedstawiający komponenty w przestrzeni dwuwymiarowej, z kolorami odpowiadającymi klastrom.')
    st.write('Krzyże oznaczają środki klastrów.')

    #Scatterplot
    fig4, ax = plt.subplots(figsize=(10, 6))
    plt.scatter(samples_pca[:, 0], samples_pca[:, 1], c=kmeans.labels_, cmap='viridis', label='Komponenty')
    plt.scatter(centers_pca[:, 0], centers_pca[:, 1], marker='x', c='red', label='Środki klastrów')
    for cluster, components in components_in_clusters.items():
        for component in components:
            plt.text(samples_pca[component_keys.index(component), 0] + 0.001, samples_pca[component_keys.index(component), 1] + 0.001, str(component), fontsize=9)
    plt.xlabel('Składowa PCA 1')
    plt.ylabel('Składowa PCA 2')
    plt.title('PCA - wizualizacja środków klastrów')
    plt.legend()
    st.pyplot(fig4)

     
if __name__ == "__main__":
    main()
