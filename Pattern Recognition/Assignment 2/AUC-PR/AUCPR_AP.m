data = csvread('data.csv');
label = data(:, 1);
score = data(:, 2);
TP = zeros(10,1);
FP= zeros(10,1);
FN = zeros(10,1);
AUC_PR = zeros(10,1);
AP = zeros(10,1);
Result = zeros(10,6);
for i = 1:10
    for j = 1:i
       if(label(j) == 1) %True Positive
           TP(i,1) = TP(i,1) + 1;
       else %False Positive
           FP(i,1) = FP(i,1) + 1;
       end
    end
    for j2 = i+1:10
        if(label(j2) == 1)
            FN(i,1) = FN(i,1) + 1;
        end
    end
end
Precision = TP./(TP+FP);
Recall = TP./(TP+FN);
AUC_PR(1,1) = (Recall(1,1)-0).*(Precision(1,1)+1)/2;
AP(1,1) = (Recall(1,1) - 0).*Precision(1,1);
for i = 2:10
    AUC_PR(i,1) = (Recall(i,1)-Recall(i-1,1)).*(Precision(i,1)+Precision(i-1,1))/2;
    AP(i,1) = (Recall(i,1)-Recall(i-1,1)).*Precision(i,1);
end
AUC_PR_SUM = sum(AUC_PR);
AP_SUM = sum(AP);
if 0
Result(:,1) = label;
Result(:,2) = score;
Result(:,3) = Precision;
Result(:,4) = Recall;
Result(:,5) = AUC_PR;
Result(:,6) = AP;
csvwrite('Q3Result.csv',Result);
end