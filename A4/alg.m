clear all
clc
E=csvread('example1.dat'); % example2.dat
algorithm(E);

function [clusters, L]=algorithm(E)

    % step 1: visualize the graph, form the adjacency matrix
    col1=E(:,1);
    col2=E(:,2);
    max_ids = max(max(col1,col2));
    As= sparse(col1, col2, 1, max_ids, max_ids);
    A= full(adjacency(graph(As))); 
    
    figure(1);
    subplot(1,2,1);
    h = plot(graph(A),'Layout','force');
    title('Graph');
    
    subplot(1,2,2);
    spy(A);
    title('Adjacency matrix');
    saveas(gcf,'graph.jpg');

    % step 2: compute the normalized Laplacian
    D=diag(sum(A,2));
    L=(D^(-1/2)*A*D^(-1/2));

    % step 3: find the k largest eigenvectors of L
    [VV, DD]=eigs(L);
    [lam, index] = sort(diag(DD),'descend');
    VV = VV(:, index);
    k = findk(lam);
    
    figure(2);
    subplot(1,2,1);
    plot(lam,'-*');
    title('Eigenvalues of normalized Laplacian');
    legend(['k = ', num2str(k), ' has the largest eigengap']);

    X = VV(:,1:k);
      
    % step 4: normalize the rows of X
    Y = X./sum(X.*X,2).^(1/2);
    
    % step 5: perform k-means
    clusters = kmeans(Y,k);
    
    % step 6: assign nodes to clusters 
    cluster_colors=hsv(k);
    for i=1:k
        cluster_members=find(clusters==i);
        highlight(h, cluster_members , 'NodeColor', cluster_colors(i,:));
        saveas(1,'cluster.jpg');
    end

    Lap = D - A;
    [FV,~] = eigs(Lap,2,'SA');
    subplot(1,2,2);
    plot(sort(FV(:,2)),'-x');
    title('Components of the Fiedler Vector');
    saveas(gcf,'spectra.jpg');
end


function k=findk(lam) % find the k that has the largest eigengap
    max_diff = 0;
    k=1;
    for i=2:length(lam) 
        diff = abs(lam(i)-lam(i-1));
        if(diff>max_diff)
            max_diff = diff;
            k = i;
        end
    end
end