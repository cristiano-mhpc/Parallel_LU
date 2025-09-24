/*This code implements LU factorization of matrix using the BLOCK ALgorithm
in order to implement the factorization in parallel. Two processes are used and
as many logical threads as possible. To create and manage threads, I used the
Boost.Thread Library. To implement application level parallelization,
I used Boost.MPI.
1. A_00 = L_00 U_00
2. A_10 = L_10 U_00
3. A_01 = L_00 U_01
4. A_11 = L_10 U_01 + L_11 U_11


Steps 2 and 3 can be done in parallel by two processes. The matrix multiplication in
the matrix update in Step 4 is offloaded to the GPU.

*/

#include <boost/thread.hpp>
#include <string>
#include <boost/lexical_cast.hpp>
#include <random>

//for parallel implementation
#include <boost/mpi.hpp>

//header files for including the kernel
#include "kernel.h"
#include "dev_array.h"
#include <math.h>


//Boost.Serialization headers
#include <boost/serialization/serialization.hpp>
#include <boost/serialization/array.hpp>
#include <boost/serialization/version.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/array.hpp>
#include <stdio.h> // used in creating File object

//Boost.Ublas headers
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/triangular.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp>
#include <boost/numeric/ublas/vector_proxy.hpp>
#include <boost/numeric/ublas/lu.hpp>
#include <boost/numeric/ublas/io.hpp>

#include <algorithm>
#include <iomanip>
#include <chrono>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdlib.h>
#include <time.h>
#include <cmath>

using namespace boost::archive;
using namespace boost::serialization;
using namespace boost::numeric::ublas;

namespace mpi = boost::mpi;

int i,j,k,m,n;
const int l = 50, b =19; // l*(b+1)  = d+1
const int d = l*(b+1) - 1;

matrix<float > Pr(b+1, b+1);
triangular_matrix<float, unit_lower> tul(b+1, b+1);
triangular_matrix<float, lower> tup(b+1, b+1);
matrix<float> P(d+1, d+1);
vector<float> moments(d+1);

void u_solver(boost::numeric::ublas::vector<float > mvs, int row , int colm) // solves for U_01 column per column
{
    mvs = matrix_vector_slice<matrix<float > >  (P, slice(row, 1, b+1), slice( colm,  0, b + 1));

    inplace_solve(tul, mvs, unit_lower_tag());

    matrix_vector_slice<matrix<float > > (P, slice(row, 1, b+1), slice(colm, 0, b+1)) = mvs; //updates P
}

void l_solver(boost::numeric::ublas::vector<float > mvs ,int roww, int colm) // solves for L_10 row per row
{
    mvs = matrix_vector_slice<matrix<float > > (P, slice(roww, 0, b+1), slice(colm, 1, b+1));

    inplace_solve (tup, mvs, lower_tag ());

    matrix_vector_slice<matrix<float > > (P, slice(roww, 0, b+1), slice(colm, 1, b+1)) = mvs; //updates P
}


void up_tri() // for obtaining the upper triangular stored in Pr
{
    for(int i = 0; i < tup.size1(); ++i)
    {
        for(int j = 0; j <  i+1 ; ++j)
        {
            tup(i,j) = Pr(j,i);
        }
    }

}

void low_tri() // for obtaining the unit lower triangular stored in Pr
{
    for(int i = 0; i < tul.size1(); ++i)
    {
        for(int j = 0; j < i ; ++j)
        {
            tul(i,j) = Pr(i,j);
        }
    }
}

int main()
{
    using namespace std::chrono;
    mpi::environment env;
    mpi::communicator world;

    int num_process = 2; //specify the number of processes to be used

    int num_threads = boost::thread::hardware_concurrency(); /* gets the maximum number of physically available
                                                                logical threads. */

    //random number generator
    std::uniform_real_distribution<float > unif(1,5);
    std::default_random_engine re;

//----------------------------populate moments(i) with random float-----------------------//


    //populate and print it
  
    std::ofstream outfile1;
    outfile1.open("moments.txt",std::ios_base::out);

    for (i = 0; i < moments.size(); ++i)
    {
    	moments(i) = unif(re);
        outfile1 << std::scientific <<moments(i) << std::endl;
    }

    outfile1.close();
    

//-----------------------------Populate P(n,m) with random float---------------------------//

    //populate and print P
    
    std::ofstream outfile2;
    outfile2.open("matrix_P.txt",std::ios_base::out);
    for ( n = 0; n < P.size1() ; n++ )
    {
        for ( m = 0; m < P.size2(); m++ )
        {
             P(n,m) = unif(re);
	         outfile2 << std::scientific << P(n,m) << std::endl;
        }
    }

    outfile2.close();
    
    //------------------begin LU factorization here----------------------------

    if (world.rank() == 0) // l rounds of Block LU factorization to perform
    {

        auto start = high_resolution_clock::now();

        for ( n = 0; n < l ; ++n )
        {

            permutation_matrix<std::size_t> PI( b+1 );//Records the row indices placed on top in the partial pivoting during LU factorization of Pr.

            Pr = matrix_range<matrix<float > > ( P, range( n*(b+1), ( n+1 )*( b+1 ) ), range( n*( b+1 ), ( n+1 )*( b + 1 )  ) );



            lu_factorize(Pr,PI); //unit lower and upper triangular factors of Pr are now stored in Pr
            world.send(1, 16, Pr); //send Pr to process 1

            /*row swapping was made in the submatrix Pr, we do the same for the
            corresponding elements of the  moments(i) and rows of P(i,j) adjacent with Pr. Row swaps made in Pr
            are encoded in PI */

            for (i = 0; i < PI.size(); ++i )
            {
                std::swap(moments(i + n*(b+1)), moments( PI(i) + n*(b+1) )); //swap corresponding rows of moments

                for (j = (n+1)*(b+1); j < P.size2(); ++j)
                {
                    std::swap(row(P, i  + n*(b+1))(j), row(P, PI(i) + n*(b+1))(j)); //swap row elements (not rows) of P to the right of Pr, j specifies the column
                }
                for (j = 0; j < n*(b+1); ++j)
                {
                    std::swap(row(P, i + n*(b+1))(j), row(P, PI(i) + n*(b+1))(j));//swap row elements (not rows) of P to the left of Pr
                }
            }

            //--------------------------------------------------------------------------------------------------//

            //update the relevant submatrix of P with the factorizations of Pr

            matrix_range<matrix<float > > (P, range(n*(b+1), (n+1)*(b+1)), range(n*(b+1), (n+1)*(b+1))) = Pr;

            //--------------------------------------------------------------------------------------------------//

            boost::thread t7(up_tri); // obtain the upper triangular factor stored in Pr
            t7.join();

            //matrix_range<matrix<float > > U_01( P, range( Start , Start + ( b + 1 )  ), range( Start + ( b+1 ) , P.size2() ) ); // if not allowed

            matrix<float > A_01( b+1, P.size2() - (n+1)*(b+1) );

            //matrix<float > U_01;

            A_01 = matrix_range<matrix<float > > ( P, range( n*(b+1), ( n+1 ) * ( b + 1 )), range( ( n+1) * ( b+1 ), P.size2() ) );

            world.send(1, 17, A_01); // send A_01 to process 1

            //-----------process 1 will proceed to compute U_01-----------------//

            //----------------Here we compute L_10 -----------------------------//

            int batch = (int) floor( ( P.size2() - (n+1)*(b+1))  / num_threads );

            int Start = n*(b+1); /*defines starting points for column updates in L_10
                                and row updates in L_10. */

            boost::numeric::ublas::vector<float > mhs(b+1);

            for (k = 0; k < batch; ++k)
            {
                boost::thread t[num_threads];

                for (i = 0 ; i < num_threads; ++i)
                {
                    int row = (n+1)*(b+1) + i + (k*num_threads);
                    t[i] = boost::thread{l_solver, mhs, row, Start};
                }

                for (int i = 0; i < num_threads; ++i)
                {
                    t[i].join();
                }

            }

            //last batch
            boost::thread t4[P.size2() - (n+1)*(b + 1) - (batch*num_threads)];
            for ( i = 0; i < P.size2() - (n+1)*(b + 1) - (batch*num_threads); ++i )
            {
                int row = i + (n+1)*(b + 1) + (batch*num_threads);

                t4[i] =boost::thread{l_solver, mhs, row, Start};
            }

            for ( int i = 0; i < P.size2() - (n+1)*(b + 1) - (batch*num_threads); ++i )
            {
                t4[i].join();
            }

            //-----------------solve for A'_11 and replace the current trailing sub matrix-----------------------------//

            matrix<float > U_01;

            //U_01 = matrix_range<matrix<float > > ( P, range(n*(b+1), ( n+1 )*( b+1 )), range(( n+1) * ( b+1 ), P.size2() ) );

            world.recv(1, 18 , U_01);// receive U_01 from process 1

            //then assign to the appropriate matrix block
            matrix_range<matrix<float > > ( P, range( n*(b+1), ( n+1 )*( b+1 ) ), range( ( n+1 ) * ( b+1 ), P.size2() ) ) = U_01;



            //------------GPU OFFLOADING TO OBTAINING A'11--------------------------------//
            int N = P.size2()-(n+1)*(b+1); // depends on the stage, n, of the iteration
            int M = b+1; //always regardles of the loop iteration
            int SIZE = N*M;


            // Allocate memory on the host
            std::vector<float> h_A(SIZE);// h_A is L10
            std::vector<float> h_B(SIZE);// h_B is U01
            std::vector<float> h_C(N*N);// h_C is A11
            std::vector<float> h_C_copy(N*N);// for checking

            // Initialize matrices on the host in a row-major fashion! Source the elements from P.

            //h_A is N by M is L10
            for (int i=0; i<N; ++i){
                for (int j=0; j<M; ++j){
                    h_A[i*M+j] = P( (n + 1)*(b + 1) + i , (n)*(b+1) + j ); //ROW MAJOR INITIALIZATION!
                }
            }

            //h_B is M by N is U01
            for (int i=0; i<M; ++i){
                for (int j=0; j<N; ++j){
                    h_B[i*N+j] = P( (n)*(b + 1) + i , (n+1)*(b+1) + j ); //ROW MAJOR INITIALIZATION!
                }
            }


            // Allocate memory on the device
            dev_array<float> d_A(SIZE);
            dev_array<float> d_B(SIZE);
            dev_array<float> d_C(N*N);


            //Copy the data from host array to device array
            d_A.set(&h_A[0], SIZE);
            d_B.set(&h_B[0], SIZE);


            //Call the kernel
            matrixMult(d_A.getData(), d_B.getData(), d_C.getData(), N, M);
            cudaDeviceSynchronize();

            //copy the result
            d_C.get(&h_C[0], N*N);
            cudaDeviceSynchronize();

            /* At this point, we have the product L10*U01, we proceed to compute A'_11. We update P with this.
	    The arrays are linearized in a row-major fashion. A for loop will be needed for this.
            Element by element update coz h_C is linearized*/

            for (i = 0; i < N; ++i){
                for (j=0; j < N; ++j){
                     P( (n+1)*(b+1) + i, (n+1)*(b+1) + j) = P( (n+1)*(b+1) + i, (n+1)*(b+1) + j) - h_C[ (i * N )+ j];
                }
            }

            //deallocate memory

            PI.clear();
            U_01.clear();
            A_01.clear();
            Pr.clear();
            tup.clear();



	    h_A.clear();
	    h_B.clear();
	    h_C.clear();


        }

        auto stop = high_resolution_clock::now();

        auto duration = duration_cast<seconds>(stop - start);

        std::ofstream outfile5;

     	outfile5.open("times.txt", std::ios_base::out);

        outfile5 << duration.count() << " seconds to factorize with GPU the matrix with dimension " << d+1 << std::endl;

       

        //---------------------------------P has now been factored-----------------------------------------//


	/*Constants is used as both input and output container. The system
 	 Ax = (LU)x = b. Let Ux = x' so Lx' = b. Solve this by forward substitution.
	 Then Solve Ux = x' by backward substitution. */

        vector<float > Constants(d+1);
        Constants = moments;
        permutation_matrix<size_t> PI3(d+1);
        lu_substitute(P, PI3, Constants);


        //print out the constant
        std::ofstream outfile1;
        outfile1.open("Constant.txt", std::ios_base::out);

        for(i = 0; i <Constants.size();++i)
        {
            outfile1 << std::scientific << Constants(i) << std ::endl;
        }


        std::ofstream outfile3;

        return 0;

    }

    if (world.rank() == 1)
    {
        for ( n = 0; n < l ; ++n )
        {
            //matrix<float > U_01 ( b+1, P.size2() - (n+1)*(b+1) );

            matrix<float > U_01;

            int batch = (int) floor( ( P.size2() - (n+1)*(b+1))  / num_threads);// the number of batches of threads to use

            int Start = n*(b+1); /*defines starting points for column updates in U_01
                                and row updates in L_10. */

            //matrix_range<matrix<float > > U_01( P, range( Start , Start + ( b + 1 )  ), range( Start + ( b+1 ) , P.size2() ) );

            world.recv(0, 16, Pr); // receive Pr from process 0
            world.recv( 0, 17, U_01 ); //receive A_01  from process 0 and store it in U_01
            boost::thread t6(low_tri); // obtain the unit lower triangular factor stored in Pr
            t6.join();

            matrix_range<matrix<float > > ( P, range( n*(b+1), ( n+1 )*( b+1 ) ), range( ( n+1) * ( b+1 ), P.size2()) ) = U_01;

            boost::numeric::ublas::vector<float > mhs(b+1);
            //Solve for U_01 block column by column. Each thread gets ones column.
            for (k = 0; k < batch; ++k)
            {
                boost::thread t[num_threads];
                for (i = 0 ; i < num_threads; ++i)
                {
                    int colm = (n+1)*(b+1) + i + (k*num_threads);

                    t[i] =boost::thread{u_solver, mhs, Start, colm }; // pass the m
                }

                for (int i = 0; i < num_threads; ++i)
                {
                    t[i].join();
                }
            }

            //last batch
            boost::thread t[ P.size2() - ( n+1 )*(b + 1) - (batch*num_threads) ];

            for (i = 0; i < P.size2() - (n+1)*(b + 1) - (batch*num_threads); ++i)
            {
                int colm = i + (n+1)*(b + 1) + (batch*num_threads);

                t[i] =boost::thread{u_solver, mhs, Start, colm};
            }

            for (int i = 0; i < P.size2() - (n+1)*(b + 1) - (batch*num_threads); ++i)
            {
                t[i].join();
            }

            U_01 = matrix_range<matrix<float > > ( P, range( n*(b+1), ( n+1 )*( b+1 ) ), range( ( n+1) * ( b+1 ), P.size2()) );

            //send back updated U_10 to process 0

            world.send( 0, 18, U_01 );

            U_01.clear();
            Pr.clear();
            tul.clear();

        }

    }

    return 0;

}
