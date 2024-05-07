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


#Creating radius neighbourhood graph
def create_graph(dataframe, radius, celltype=False):
    if celltype == False:
        df = dataframe
    else:
        df = dataframe[dataframe["celltype"] == celltype]
    coordinates_list = df[['nucleus.x', 'nucleus.y']].values
    info_list = df[['cell.ID', 'celltype']].values
    graph = radius_neighbors_graph(coordinates_list, radius, mode='distance', metric='minkowski', p=2, metric_params=None, include_self=False)
    return graph, coordinates_list, info_list


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

    #Scatterplot with all celltypes
    fig, ax = plt.subplots()
    sns.scatterplot(x="nucleus.x", y="nucleus.y", data=merged_df, hue="celltype")
    ax.set_xlim(min_x*0.98, max_x*1.02)
    plt.title('Scatterplot of cell data')
    plt.xlabel("x_column")
    plt.ylabel("y_column")
    plt.legend(title="Cell type")
    plt.grid(True)
    st.pyplot(fig)


    #Creating scatterplot with selectetypes
    fig, ax = plt.subplots()

    # Funkcja generująca scatterplot dla wybranych typów komórek
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
        plt.legend(title="Cell type")
        plt.grid(True)
        st.pyplot(fig)

    #Selec
    selected_cell_types = st.sidebar.multiselect("Select cell types", merged_df["celltype"].unique())

    # Generowanie scatterplotu dla wybranych typów komórek
    generate_scatterplot(selected_cell_types)

    #Creating radius neighbourhood graphs for all cells and only bcells
    radius = st.sidebar.slider('Select radius:', min_value=1, max_value=50, value=30)
    graph_bcell, coordinates_bcell, info_bcell = create_graph(merged_df, radius, "Bcell")
    graph_all, coordinates_all, info_all= create_graph(merged_df, radius)
    
    #Finding connected components from bcells 
    n_components_bcell, labels_bcell = connected_components(csgraph=graph_bcell, directed=False, return_labels=True)

    #Counting how many cells are in components and filtering out those with less than 15
    counts_bcell = np.bincount(labels_bcell)
    threshold = 15
    indices = np.where(counts_bcell > threshold)[0]
    
    cells_dict ={} 
    for index in indices: 
        cells_dict[int(index)] = np.where(labels_bcell == index)[0]


    slownik = {}
    for component in cells_dict.keys():
        slownik[component] = []
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
                slownik[component].append(tuple(info_all[index])) 
        #Creating the dict where the key is the index of component and the value is the list of types of cells        
        slownik[component] = list(set(slownik[component]))

    # Tworzenie wykresów słupkowych dla każdej składowej
    for component in slownik.keys(): 
        celltypes = [celltype for index, celltype in slownik[component]]
        total_cells = len(celltypes)
        unique_celltypes = list(set(celltypes))
        counts = [celltypes.count(celltype) / total_cells for celltype in unique_celltypes]  # Obliczanie procentowego udziału
        fig2, ax2 = plt.subplots(figsize=(10, 6))  # Ustawienie rozmiaru figury na 10x6 cali
        plt.bar(unique_celltypes, counts, color='pink')  # Użycie funkcji plt.bar() do stworzenia wykresu słupkowego
        plt.title(f"Percentage of each celltype in component {component}")
        plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
        plt.xlabel('Celltype')
        plt.ylabel('Percentage')
        plt.xticks(rotation=30, ha='center')
        st.pyplot(fig2)
            
if __name__ == "__main__":
    main()

