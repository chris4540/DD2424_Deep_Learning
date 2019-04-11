module ann_f
contains
function softmax(x, axis) result(ret)
    implicit none
    real(kind=8), dimension(:, :), intent(in) :: x
    integer(kind=4), intent(in), optional :: axis
    real(kind=8), dimension(size(x,1), size(x,2)) :: ret
    ! local var
    real(kind=8), dimension(size(x,1), size(x,2)) :: exp_prob
    integer(kind=4) :: axis_
    integer :: i

    if(present(axis))then
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
    ! integer(kind=4) :: i

    ret = matmul(W_mat, X_mat)

    ! Another implementation
    ret = matmul(W_mat, X_mat) + SPREAD(reshape(b_vec, [size(W_mat,1)]), 2, size(X_mat,2))
    ! do i = 1, ubound(ret, 1)
    !     ret(:, i) = ret(:, i) + b_vec(:, 1)
    ! end do

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
    integer :: n_data
    integer :: n_class
    ! ======================================================
    n_data = size(X_mat,2)
    n_class = size(W_mat,1)
    p_mat = evaluate_classifier(X_mat, W_mat, b_vec)

    g_mat = -(Y_mat - p_mat)

    grad_b = reshape(sum(g_mat, dim=2), [n_class, 1])
    grad_b = grad_b / n_data

    ! grad_W = g_mat.dot(X_mat.T) / n_data
    grad_W = matmul(g_mat, TRANSPOSE(X_mat)) / n_data
    ! grad_W += 2 * lambda_ * W_mat
    grad_W = grad_W + 2 *lambda_ * W_mat
end subroutine
end module
