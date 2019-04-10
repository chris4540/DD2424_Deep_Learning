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

! function evaluate_classifier(X_mat, W_mat, b_vec) result(ret)
!     real(kind=8), dimension(:, :), intent(in) :: X_mat
!     real(kind=8), dimension(:, :), intent(in) :: W_mat
!     real(kind=8), dimension(:, :), intent(in) :: b_vec
!     real(kind=8), dimension(size(W_mat,1), size(X_mat,2)) :: ret
!     INTERFACE
!         FUNCTION softmax(inarg, axis) result(ret)
!             real(kind=8), dimension(:, :), intent(in) :: inarg
!             integer(kind=4), intent(in),optional :: axis
!             real(kind=8), dimension(size(inarg,1), size(inarg,2)) :: ret
!         END FUNCTION  softmax
!     END INTERFACE


!     ret = softmax(ret, 0)
! end function evaluate_classifier

function evaluate_classifier(X_mat, W_mat, b_vec) result(ret)
    implicit none
    real(kind=8), dimension(:, :), intent(in) :: X_mat
    real(kind=8), dimension(:, :), intent(in) :: W_mat
    real(kind=8), dimension(:, :), intent(in) :: b_vec
    real(kind=8), dimension(size(W_mat,1), size(X_mat,2)) :: ret
    INTERFACE
        FUNCTION softmax(inarg, axis) result(ret)
            real(kind=8), dimension(:, :), intent(in) :: inarg
            integer(kind=4), intent(in),optional :: axis
            real(kind=8), dimension(size(inarg,1), size(inarg,2)) :: ret
        END FUNCTION  softmax
    END INTERFACE
    ! ===============================================================
    integer(kind=4) :: i

    ret = matmul(W_mat, X_mat)

    ! Another implementation
    ! ret = matmul(W_mat, X_mat) + SPREAD(reshape(b_vec, [size(W_mat,1)]), 2, size(X_mat,2))
    do i = 1, ubound(ret, 1)
        ret(:, i) = ret(:, i) + b_vec(:, 1)
    end do

    ret = softmax(ret, 0)
end function evaluate_classifier

