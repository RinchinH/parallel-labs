#include <mpi.h>
#include <vector>
#include <random>
#include <cmath>
#include <iostream>
#include <string>
#include <stdexcept>
#include <cstdio>
using std::vector; using std::string;

struct Args{ long N=2000; double eps=1e-8; int maxit=10000; unsigned seed=42; bool quiet=true; };
static Args parse(int argc,char**argv){
    Args a; for(int i=1;i<argc;i++){ string s=argv[i];
        auto need=[&](const char* n){ if(++i>=argc) throw std::runtime_error(n); return string(argv[i]); };
        if(s=="-n"||s=="--size") a.N=std::stol(need("--size"));
        else if(s=="-eps"||s=="--eps") a.eps=std::stod(need("--eps"));
        else if(s=="-k"||s=="--maxit") a.maxit=std::stoi(need("--maxit"));
        else if(s=="-s"||s=="--seed") a.seed=(unsigned)std::stoul(need("--seed"));
        else if(s=="-q"||s=="--quiet") a.quiet=true;
    } return a;
}
struct Block{ long beg,cnt; };
static Block split(long N,int P,int r){ long q=N/P, rem=N%P; long cnt=q+(r<rem?1:0); long beg=r*q+std::min<long>(r,rem); return {beg,cnt}; }
static double l2sq_local(const vector<double>& v){ double s=0; for(double x:v) s+=x*x; return s; }

int main(int argc,char**argv){
    MPI_Init(&argc,&argv);
    int rank,size; MPI_Comm_rank(MPI_COMM_WORLD,&rank); MPI_Comm_size(MPI_COMM_WORLD,&size);

    // --- глушим stdout у всех, кроме rank 0 (чтобы не дублировать строки) ---
    if(rank!=0){ (void)freopen("/dev/null","w",stdout); }

    Args arg=parse(argc,argv); const long N=arg.N;

    Block blk=split(N,size,rank); const long nloc=blk.cnt;
    vector<double> x_true_loc(nloc), b_loc(nloc), x_loc(nloc,0.0), x_new(nloc,0.0);

    // рассылаем x_true
    vector<int> counts(size), displs(size);
    for(int r=0;r<size;r++){ Block t=split(N,size,r); counts[r]=(int)t.cnt; displs[r]=(int)t.beg; }
    vector<double> x_true_all; double sum_xt=0.0;
    if(rank==0){
        x_true_all.resize(N);
        std::mt19937 rng(arg.seed);
        std::uniform_real_distribution<double> U(-1.0,1.0);
        for(long i=0;i<N;i++){ x_true_all[i]=U(rng); sum_xt+=x_true_all[i]; }
    }
    MPI_Scatterv(rank==0?x_true_all.data():nullptr, counts.data(), displs.data(), MPI_DOUBLE,
                 x_true_loc.data(), (int)nloc, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&sum_xt,1,MPI_DOUBLE,0,MPI_COMM_WORLD);

    // b = (2N I + J) x_true
    for(long i=0;i<nloc;i++) b_loc[i]=2.0*double(N)*x_true_loc[i]+sum_xt;

    // ||b||
    double nb2_loc=l2sq_local(b_loc), nb2=0.0; MPI_Allreduce(&nb2_loc,&nb2,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
    const double norm_b=std::sqrt(nb2);

    MPI_Barrier(MPI_COMM_WORLD);
    double t0=MPI_Wtime(); int it=0; double residual=1e300;

    // Якоби для A = 2N*I + J
    while(it<arg.maxit){
        double sum_loc=0.0; for(double v: x_loc) sum_loc+=v;
        double sum_x=0.0;   MPI_Allreduce(&sum_loc,&sum_x,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
        for(long i=0;i<nloc;i++){
            double sum_except_i = sum_x - x_loc[i];
            x_new[i] = (b_loc[i] - sum_except_i) / (2.0*double(N)+1.0);
        }
        x_loc.swap(x_new);

        double r2_loc=0.0;
        for(long i=0;i<nloc;i++){ double s = 2.0*double(N)*x_loc[i] + sum_x; double ri = s - b_loc[i]; r2_loc += ri*ri; }
        double r2=0.0; MPI_Allreduce(&r2_loc,&r2,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
        residual = std::sqrt(r2)/norm_b;
        ++it; if(residual<arg.eps) break;
    }
    MPI_Barrier(MPI_COMM_WORLD);
    double t1=MPI_Wtime();

    // относительная ошибка к x_true
    double e2_loc=0.0, xt2_loc=0.0;
    for(long i=0;i<nloc;i++){ double d=x_loc[i]-x_true_loc[i]; e2_loc+=d*d; xt2_loc+=x_true_loc[i]*x_true_loc[i]; }
    double e2=0.0, xt2=0.0; MPI_Allreduce(&e2_loc,&e2,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
    MPI_Allreduce(&xt2_loc,&xt2,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
    double rel_err = std::sqrt(e2)/std::sqrt(xt2);

    if(rank==0){
        if(arg.quiet) std::cout<<(t1-t0)<<","<<it<<","<<residual<<","<<rel_err<<"\n";
        else std::cout<<"Finished "<<(t1-t0)<<" s it="<<it<<" r="<<residual<<" rel_err="<<rel_err<<"\n";
    }
    MPI_Finalize(); return 0;
}
