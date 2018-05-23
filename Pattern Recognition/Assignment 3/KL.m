function kl = KL(A, B)
kl = A(1)*log2(A(1)/B(1))+A(2)*log2(A(2)/B(2));
end
