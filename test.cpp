#include<iostream>

using namespace std;

int n;

int solve(int n) {
    if (n == 1) {
        return 1;
    }
    return n * solve(n - 1);
}

void swap(int &a, int &b) {
    int temp = b;
    b = a;
    a = temp;
}

int main() {

    // cin>>n;
    // cout<<solve(n)<<endl;
    int a = 5;
    int b = 10;
    swap(a, b);
    cout<<a<<" "<<b<<endl;

    return 0;
}