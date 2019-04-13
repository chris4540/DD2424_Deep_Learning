module ann_for
    ! Experimental f2py testing for rewriting the ann.py to fortran
    ! Python + numpy use performs better in matrix multiplication than fortran
    ! Using np.asfortranarray can avoid copying due to C-layout or fortran layout
    !
    ! Freeze this project
    !
    ! Testing CPU:
    !   Intel(R) Celeron(R) CPU  N2840  @ 2.16GHz
    ! Testing OS:
    !   Linux 4.15.0-42-generic
    ! Testing Compiler:
    !   GNU Fortran (Ubuntu 7.3.0-27ubuntu1~18.04) 7.3.0
    ! ==========================================================================
    ! Testing results:
    !
    ! evaluate_classifier:
    !   >>> ann_py.evaluate_classifier(X_mat, W_mat, b_vec)
    !   244 ms ± 26.3 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
    !   >>> ann_f.evaluate_classifier(X_mat_f, W_mat_f, b_vec_f)
    !   436 ms ± 3.46 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
    !   >>> X_mat.shape
    !   (3072, 10000)
    !
    ! softmax:
    !   >>> ann_py.softmax(s_mat)
    !   13.3 ms ± 688 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)
    !   >>> ann_f.softmax(s_mat)
    !   7.93 ms ± 75.7 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)
contains
function softmax(x, axis) result(ret)
    implicit none
    real(kind=8), dimension(:, :), intent(in) :: x
    integer(kind=4), intent(in), optional :: axis
    real(kind=8), dimension(size(x,1), size(x,2)) :: ret
    ! local variable
    real(kind=8), dimension(size(x,1), size(x,2)) :: exp_prob
    integer(kind=4) :: axis_
    integer(kind=4) :: i  ! an index iterator

    if (present(axis)) then
        axis_ = axis
    else
        axis_ = 0
    endif
    ! ========================================================
    exp_prob = exp(x)

    if (axis_ == 0) then
        ! follow the python convension
        do i = 1, ubound(exp_prob, 2)
            ret(:, i) = exp_prob(:, i) / sum(exp_prob(:, i))
        end do
    else
        do i = 1, ubound(exp_prob, 1)
            ret(i, :) = exp_prob(i, :) / sum(exp_prob(i, :))
        end do
    end if
end function softmax

function evaluate_classifier(X_mat, W_mat, b_vec) result(ret)
    implicit none
    real(kind=8), dimension(:, :), intent(in) :: X_mat
    real(kind=8), dimension(:, :), intent(in) :: W_mat
    real(kind=8), dimension(:, :), intent(in) :: b_vec
    real(kind=8), dimension(size(W_mat,1), size(X_mat,2)) :: ret
    ! ===============================================================
    ret = matmul(W_mat, X_mat) + SPREAD(reshape(b_vec, [size(W_mat,1)]), 2, size(X_mat,2))
    ret = softmax(ret, 0)
end function evaluate_classifier

function compute_cost(X_mat, Y_mat, W_mat, b_vec, lambda_) result(ret)
    implicit none
    real(kind=8), dimension(:, :), intent(in) :: X_mat
    real(kind=8), dimension(:, :), intent(in) :: W_mat
    real(kind=8), dimension(:, :), intent(in) :: Y_mat
    real(kind=8), dimension(:, :), intent(in) :: b_vec
    real(kind=8)                 , intent(in) :: lambda_
    real(kind=8)                              :: ret
    ! ===================================================
    ! local vars
    real(kind=8), dimension(size(W_mat,1), size(X_mat,2)) :: p_mat
    real(kind=8), dimension(size(X_mat,2)) :: cross_entro

    p_mat = evaluate_classifier(X_mat, W_mat, b_vec)
    cross_entro = sum(Y_mat * p_mat, dim=1)
    cross_entro = -log(cross_entro)

    ret = (sum(cross_entro) / size(X_mat,2)) + lambda_*sum(W_mat**2)
end function compute_cost

subroutine compute_gradients(X_mat, Y_mat, W_mat, b_vec, lambda_, grad_W, grad_b)
    implicit none
    real(kind=8), dimension(:, :), intent(in) :: X_mat
    real(kind=8), dimension(:, :), intent(in) :: W_mat
    real(kind=8), dimension(:, :), intent(in) :: Y_mat
    real(kind=8), dimension(:, :), intent(in) :: b_vec
    real(kind=8)                 , intent(in) :: lambda_
    real(kind=8), dimension(size(W_mat,1), size(W_mat,2)), intent(out) :: grad_W
    real(kind=8), dimension(size(b_vec,1), size(b_vec,2)), intent(out) :: grad_b
    ! ======================================================
    real(kind=8), dimension(size(W_mat,1), size(X_mat,2)) :: p_mat
    real(kind=8), dimension(size(W_mat,1), size(X_mat,2)) :: g_mat
    integer(kind=4) :: n_data
    integer(kind=4) :: n_class
    integer(kind=4) :: n_dim
    real(kind=8) :: alpha
    real(kind=8) :: beta
    ! ======================================================
    n_dim = size(X_mat,1)
    n_data = size(X_mat,2)
    n_class = size(W_mat,1)
    p_mat = evaluate_classifier(X_mat, W_mat, b_vec)

    g_mat = -(Y_mat - p_mat)

    grad_b = reshape(sum(g_mat, dim=2), [n_class, 1])
    grad_b = grad_b / n_data

    ! ===============================================
    ! call blas to compute g*X'/n + 2*lambda_*W_mat
    ! grad_W = matmul(g_mat, TRANSPOSE(X_mat)) / n_data
    ! grad_W = grad_W + 2 *lambda_ * W_mat
    alpha = 1.D0 / n_data
    beta = 2.D0 * lambda_
    grad_W = W_mat
    call DGEMM(  &
     &  'N', 'T', n_class, n_dim, n_data, &
     &  alpha, g_mat, n_class, X_mat, n_dim, &
     &  beta, grad_W, n_class)
    ! ===============================================
end subroutine compute_gradients
end module ann_for
